"""
noi_tu_selfbot.py

NOTE ON THE LIBRARY: modern discord.py removed the `self_bot=True` param entirely.
If this script is meant to log in with a *user* token, you need the community fork:

    pip uninstall discord.py
    pip install -U discord.py-self

Heads up: Automating a personal Discord account is against Discord's Terms of Service 
and can get the account disabled.
"""

# -*- coding: utf-8 -*-

import os
import re
import time
import random
import asyncio
import logging
import unicodedata
from collections import defaultdict, deque
from typing import Optional, Tuple, Dict, List, Set

import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DISCORD_TOKEN              = os.getenv("DISCORD_TOKEN")
CHANNEL_ID_RAW             = os.getenv("CHANNEL_ID")
OWNER_ID                   = int(os.getenv("OWNER_ID", "0"))
GAME_MASTER_BOT_ID         = int(os.getenv("GAME_MASTER_BOT_ID", "1103932552701550622"))
REQUIRED_REACTIONS         = int(os.getenv("REQUIRED_REACTIONS", "1"))
MIN_SEND_INTERVAL          = int(os.getenv("MIN_SEND_INTERVAL", "120"))
USED_WORDS_MAXLEN          = int(os.getenv("USED_WORDS_MAXLEN", "500"))

# Dynamic history checking constraints
TRIPLE_CHECK_HISTORY_LIMIT = int(os.getenv("TRIPLE_CHECK_HISTORY_LIMIT", "1"))
BUSY_CHECK_HISTORY_LIMIT   = int(os.getenv("BUSY_CHECK_HISTORY_LIMIT", "5"))
BUSY_TYPER_THRESHOLD       = int(os.getenv("BUSY_TYPER_THRESHOLD", "3"))
PACKED_TYPER_THRESHOLD     = int(os.getenv("PACKED_TYPER_THRESHOLD", "8"))
PACKED_CHECK_HISTORY_LIMIT = int(os.getenv("PACKED_CHECK_HISTORY_LIMIT", "12"))
TYPING_WINDOW_SECONDS      = int(os.getenv("TYPING_WINDOW_SECONDS", "10"))

WORDS_FILE                 = "vietnamese_words.txt"
LOG_FILE                   = "channel_messages.log"

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def normalize(text: str) -> str:
    """Normalizes and lowercases Vietnamese text."""
    return unicodedata.normalize("NFC", text).strip().lower()


class NoiTuSelfbot:
    MAX_SYLLABLES = 2

    def __init__(self):
        if not DISCORD_TOKEN:
            raise SystemExit("DISCORD_TOKEN missing from .env")
        if not CHANNEL_ID_RAW:
            raise SystemExit("CHANNEL_ID missing from .env")

        self.token: str = DISCORD_TOKEN
        self.channel_id: int = int(CHANNEL_ID_RAW)

        # Word Dictionary states
        self.phrases: Set[str] = set()
        self.phrases_by_first_syllable: Dict[str, List[str]] = defaultdict(list)
        
        # Game states
        self.last_word: Optional[str] = None
        self.last_word_message_id: Optional[int] = None
        self.used_words: deque = deque(maxlen=USED_WORDS_MAXLEN)
        self._used_words_set: Set[str] = set()

        # Message tracking
        self.xd_messages: Set[int] = set()
        self.validated_messages: Set[int] = set()
        
        # Async Control & Timing
        self.word_ready = asyncio.Event()
        self.last_send_time: float = 0.0
        self.force_mode: Optional[str] = None
        self.typing_users: Dict[int, float] = {}

        # Bot Client
        self.client = commands.Bot(command_prefix="nt!", help_command=None)
        self.client.event(self.on_ready)
        self.client.event(self.on_message)
        self.client.event(self.on_reaction_add)
        self.client.event(self.on_typing)

    # ------------------------------------------------------------------
    # Core Internals
    # ------------------------------------------------------------------

    def _track_used(self, word: str) -> None:
        """Add word to the bounded used-words tracker efficiently."""
        if len(self.used_words) == self.used_words.maxlen:
            self._used_words_set.discard(self.used_words[0])
        self.used_words.append(word)
        self._used_words_set.add(word)

    def _log_to_file(self, message: discord.Message, prefix: str = "") -> None:
        """Helper to write a structured line to the log file."""
        content = message.content.replace("\n", " ")
        log.info("%s%s: %s", prefix, self._display_author(message), content)

    @staticmethod
    def _display_author(message: discord.Message) -> str:
        author = message.author
        disc = getattr(author, "discriminator", "0")
        return author.name if not disc or disc == "0" else f"{author.name}#{disc}"

    @staticmethod
    def _bot_x_reaction(message: discord.Message, bot_id: int) -> bool:
        """Check whether the bot has already placed an 'X' reaction."""
        return any(str(r.emoji) == "X" and r.me for r in message.reactions)

    def _active_typer_count(self) -> int:
        """Count distinct typing users, pruning stale entries automatically."""
        now = time.monotonic()
        stale = [uid for uid, ts in self.typing_users.items() if now - ts > TYPING_WINDOW_SECONDS]
        for uid in stale:
            del self.typing_users[uid]
        return len(self.typing_users)

    # ------------------------------------------------------------------
    # Word Management
    # ------------------------------------------------------------------

    async def load_words(self) -> None:
        """Loads and processes the word dictionary from file."""
        skipped = 0
        try:
            with open(WORDS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    phrase = normalize(line)
                    if not phrase:
                        continue
                    
                    syllables = phrase.split()
                    if len(syllables) != self.MAX_SYLLABLES:
                        skipped += 1
                        continue
                        
                    self.phrases.add(phrase)
                    self.phrases_by_first_syllable[syllables[0]].append(phrase)
                    
            log.info("Loaded %d 2-syllable words (skipped %d)", len(self.phrases), skipped)
        except FileNotFoundError:
            log.error("Fatal: %s not found!", WORDS_FILE)
            raise SystemExit(1)

    def get_next_word(self, last_phrase: str) -> Optional[str]:
        """Fetch a valid, ideally unused next word based on the last phrase."""
        if not last_phrase:
            return None
            
        last_phrase = normalize(last_phrase)
        syllables = last_phrase.split()
        if not syllables:
            return None

        candidates = self.phrases_by_first_syllable.get(syllables[-1], [])
        if not candidates:
            return None

        # Prioritize words we haven't used recently
        fresh = [c for c in candidates if c != last_phrase and c not in self._used_words_set]
        pool = fresh or [c for c in candidates if c != last_phrase] or candidates
        return random.choice(pool)

    def find_last_valid_phrase(self, content: str) -> Optional[str]:
        """Extracts the final N syllables and checks if they form a valid dictionary word."""
        text = normalize(content)
        tokens = re.findall(r"\w+", text)
        if len(tokens) < self.MAX_SYLLABLES:
            return None

        phrase = " ".join(tokens[-self.MAX_SYLLABLES:])
        return phrase if phrase in self.phrases else None

    def _extract_gm_started_word(self, message: discord.Message) -> Optional[str]:
        """Parse a Game Master new round announcement."""
        if "khong co trong tu dien" in message.content.lower():
            return None

        match = re.search(r"\*\*(.+?)\*\*", message.content)
        if not match:
            return None

        phrase = normalize(match.group(1))
        if len(phrase.split()) != self.MAX_SYLLABLES:
            return None
            
        return phrase

    # ------------------------------------------------------------------
    # Verification & History Logic
    # ------------------------------------------------------------------

    async def _evaluate_message(self, message: discord.Message, allow_update: bool = True) -> bool:
        """Evaluates an incoming message to see if it updates the active last_word."""
        if message.author.id == self.client.user.id:
            return False
        if message.id in self.xd_messages or message.id in self.validated_messages:
            return False
            
        if self._bot_x_reaction(message, self.client.user.id):
            self.xd_messages.add(message.id)
            return False

        total_reactions = sum(r.count for r in message.reactions)
        if total_reactions < REQUIRED_REACTIONS:
            return False

        phrase = self.find_last_valid_phrase(message.content)
        if not phrase:
            return False

        self.validated_messages.add(message.id)
        self._log_to_file(message, prefix=f"VALIDATED ({total_reactions} reactions) ")
        log.info("Message %d validated (%d reactions) -> '%s'", message.id, total_reactions, phrase)

        if allow_update:
            if self.last_word_message_id is None or message.id > self.last_word_message_id:
                self.last_word = phrase
                self.last_word_message_id = message.id
                self.word_ready.set()
                log.info("  -> last_word updated to '%s' (msg_id=%d)", phrase, message.id)
            else:
                log.info("  -> skipped (message %d is older than current last_word)", message.id)
                return False
                
        return True

    async def _get_freshest_last_word(self, channel: discord.abc.Messageable, history_limit: int) -> Tuple[Optional[str], Optional[int]]:
        """Fetch the history to ensure we reply to the absolute newest valid word."""
        try:
            async for message in channel.history(limit=history_limit):
                # Ignore self or already X'd messages
                if message.author.id == self.client.user.id or message.id in self.xd_messages:
                    continue
                    
                if self._bot_x_reaction(message, self.client.user.id):
                    self.xd_messages.add(message.id)
                    continue

                # Handle GM explicit round triggers
                if message.author.id == GAME_MASTER_BOT_ID:
                    phrase = self._extract_gm_started_word(message)
                    if phrase:
                        return phrase, message.id
                    continue

                # Standard message validation
                total_reactions = sum(r.count for r in message.reactions)
                if total_reactions >= REQUIRED_REACTIONS:
                    phrase = self.find_last_valid_phrase(message.content)
                    if phrase:
                        return phrase, message.id

                # If checking minimally, break if the immediate last message fails
                if history_limit == 1:
                    break

        except Exception as e:
            log.error("Error fetching channel history: %s", e)

        return None, None

    # ------------------------------------------------------------------
    # Loops
    # ------------------------------------------------------------------

    async def game_loop(self) -> None:
        """Main loop that respects timing intervals and fires words."""
        await self.client.wait_until_ready()
        
        while not self.client.is_closed():
            await self.word_ready.wait()
            self.word_ready.clear()
            
            if not self.last_word:
                continue

            # Throttle based on MIN_SEND_INTERVAL
            time_since_last = time.time() - self.last_send_time
            if time_since_last < MIN_SEND_INTERVAL:
                await asyncio.sleep(MIN_SEND_INTERVAL - time_since_last)

            # Determine scan depth based on channel velocity
            typers = self._active_typer_count()
            if self.force_mode == "packed" or (not self.force_mode and typers >= PACKED_TYPER_THRESHOLD):
                history_limit = PACKED_CHECK_HISTORY_LIMIT
            elif self.force_mode == "busy" or (not self.force_mode and typers >= BUSY_TYPER_THRESHOLD):
                history_limit = BUSY_CHECK_HISTORY_LIMIT
            else:
                history_limit = TRIPLE_CHECK_HISTORY_LIMIT

            channel = self.client.get_channel(self.channel_id)
            if not channel:
                log.error("Could not fetch target channel ID %s.", self.channel_id)
                continue

            fresh_word, fresh_id = await self._get_freshest_last_word(channel, history_limit)
            active_word = fresh_word if fresh_word else self.last_word
            
            next_word = self.get_next_word(active_word)
            
            if next_word:
                try:
                    await channel.send(next_word)
                    self._track_used(next_word)
                    self.last_send_time = time.time()
                    log.info("Successfully sent next word: '%s'", next_word)
                except Exception as e:
                    log.error("Failed to send word: %s", e)

    async def heartbeat_check(self) -> None:
        """Periodic logging of the bot's memory state."""
        await self.client.wait_until_ready()
        while not self.client.is_closed():
            await asyncio.sleep(30)
            log.info(
                "[heartbeat] last_word=%s | msg_id=%s | xd=%d | validated=%d",
                self.last_word,
                self.last_word_message_id,
                len(self.xd_messages),
                len(self.validated_messages),
            )

    # ------------------------------------------------------------------
    # Discord Events
    # ------------------------------------------------------------------

    async def on_ready(self) -> None:
        log.info("Logged in as %s (ID: %s)", self.client.user, self.client.user.id)
        log.info("Targeting channel ID %s", self.channel_id)
        self.client.loop.create_task(self.game_loop())
        self.client.loop.create_task(self.heartbeat_check())

    async def on_typing(self, channel: discord.abc.Messageable, user: discord.abc.User, when) -> None:
        if getattr(channel, "id", None) != self.channel_id or user.id == self.client.user.id:
            return
        self.typing_users[user.id] = time.monotonic()

    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User) -> None:
        message = reaction.message
        if message.channel.id != self.channel_id or message.author.id == self.client.user.id:
            return

        # If the bot placed an X, invalidate that message
        if user.id == self.client.user.id and str(reaction.emoji) == "X":
            self.xd_messages.add(message.id)
            self.validated_messages.discard(message.id)
            log.info("Bot X'd message %d from %s", message.id, message.author.name)
            
            if self.last_word:
                phrase = self.find_last_valid_phrase(message.content)
                if phrase and phrase == self.last_word and message.id == self.last_word_message_id:
                    self.last_word = None
                    self.last_word_message_id = None
                    log.info("Cleared last_word because bot X'd the validated message")
            return

        await self._evaluate_message(message)

    async def on_message(self, message: discord.Message) -> None:
        # Owner DM Commands overrides
        if isinstance(message.channel, discord.DMChannel) and OWNER_ID and message.author.id == OWNER_ID:
            cmd = message.content.strip().lower()
            if cmd in ["!calm", "!busy", "!packed"]:
                self.force_mode = cmd[1:]
                await message.channel.send(f"✅ Mode locked to **{self.force_mode}**.")
                log.info("Owner forced mode: %s", self.force_mode)
            elif cmd == "!auto":
                self.force_mode = None
                await message.channel.send("✅ Mode set back to **auto**.")
                log.info("Owner cleared force mode -> auto")
            elif cmd == "!mode":
                typers = self._active_typer_count()
                if self.force_mode:
                    status = f"🔒 Forced to **{self.force_mode}** (auto disabled)"
                else:
                    if typers >= PACKED_TYPER_THRESHOLD:
                        current = "packed"
                    elif typers >= BUSY_TYPER_THRESHOLD:
                        current = "busy"
                    else:
                        current = "calm"
                    status = f"🔄 Auto — currently **{current}** ({typers} typers detected)"
                await message.channel.send(status)
            return

        # Standard Channel Logic
        if message.channel.id != self.channel_id or message.author.id == self.client.user.id or message.id in self.xd_messages:
            return

        if message.author.id == GAME_MASTER_BOT_ID:
            if "khong co trong tu dien" in message.content.lower():
                log.info("GM says 'khong co trong tu dien' -- ignoring, keeping last_word='%s'", self.last_word)
                return

            phrase = self._extract_gm_started_word(message)
            if phrase:
                self.last_word = phrase
                self.last_word_message_id = message.id
                self.word_ready.set()
                log.info("New round from GM! Starting word: '%s' (msg_id=%d)", phrase, message.id)
            return

        if self._bot_x_reaction(message, self.client.user.id):
            self.xd_messages.add(message.id)
            return

        self._log_to_file(message)
        await self._evaluate_message(message)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bot = NoiTuSelfbot()
    # Load the dictionary purely synchronously before starting the bot loop
    asyncio.run(bot.load_words())
    # Start the bot connection
    bot.client.run(bot.token)
    
