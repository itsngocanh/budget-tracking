import discord
from discord.ext import commands, tasks
import json
import os
from datetime import datetime, time, timezone
import csv
import git
from dotenv import load_dotenv
import asyncio
import logging
from typing import Dict, List, Optional, Any
from functools import lru_cache
import threading
import queue
import time as time_module

# Configure logging for better performance monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/discord_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Define intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# Initialize the bot with intents
bot = commands.Bot(command_prefix="!", intents=intents)

# --- Configuration ---
TOKEN = os.getenv('DISCORD_TOKEN')
if not TOKEN:
    logger.error("DISCORD_TOKEN environment variable not set. Please set it in your .env file.")
    exit()

GUILD_ID = int(os.getenv('GUILD_ID', 1378927251877531799))
DATA_FILE = 'data/expenses.json'

# --- Configuration for daily operations ---
DAILY_MESSAGE_CHANNEL_ID = int(os.getenv('DAILY_MESSAGE_CHANNEL_ID', 1379091155538546769))
DAILY_CSV_EXPORT_CHANNEL_ID = int(os.getenv('DAILY_CSV_EXPORT_CHANNEL_ID', 1385852449063174154))
CSV_EXPORT_TIME_UTC = time(3, 0, 0, tzinfo=timezone.utc)

# --- Configuration for GitHub automation ---
LOCAL_REPO_PATH = '.'
GIT_BRANCH = 'main'

# --- Performance optimization: Global data cache ---
class DataCache:
    """Thread-safe data cache for better performance."""
    
    def __init__(self):
        self._data = None
        self._lock = threading.Lock()
        self._last_modified = 0
        self._cache_ttl = 30  # Cache for 30 seconds
    
    def get_data(self) -> Dict[str, Any]:
        """Get cached data or load from file if cache is stale."""
        current_time = time_module.time()
        
        with self._lock:
            if (self._data is None or 
                current_time - self._last_modified > self._cache_ttl or
                self._is_file_newer()):
                self._data = self._load_from_file()
                self._last_modified = current_time
        
        return self._data.copy()  # Return a copy to prevent external modifications
    
    def update_data(self, new_data: Dict[str, Any]) -> None:
        """Update cache and save to file."""
        with self._lock:
            self._data = new_data.copy()
            self._last_modified = time_module.time()
            self._save_to_file(new_data)
    
    def _is_file_newer(self) -> bool:
        """Check if the file has been modified since last cache update."""
        try:
            file_mtime = os.path.getmtime(DATA_FILE)
            return file_mtime > self._last_modified
        except OSError:
            return True
    
    def _load_from_file(self) -> Dict[str, Any]:
        """Load data from file with optimized error handling."""
        default_data = {
            "current_balance": 0.0,
            "income": [],
            "outcome": [],
            "savings": [],
            "initial_income_total": 0.0,
            "initial_outcome_total": 0.0,
            "initial_savings_total": 0.0,
            "last_summary_date": None
        }
        
        if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
            logger.info(f"'{DATA_FILE}' not found or is empty. Using default structure.")
            return default_data
        
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure all required keys exist
                for key, value in default_data.items():
                    data.setdefault(key, value)
                logger.info(f"Data loaded successfully. Current balance: {data['current_balance']}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from '{DATA_FILE}': {e}. Using default structure.")
            return default_data
        except Exception as e:
            logger.error(f"Unexpected error loading '{DATA_FILE}': {e}. Using default structure.")
            return default_data
    
    def _save_to_file(self, data: Dict[str, Any]) -> None:
        """Save data to file with optimized error handling."""
        try:
            # Create backup before saving
            if os.path.exists(DATA_FILE):
                backup_file = f"{DATA_FILE}.backup"
                try:
                    os.replace(DATA_FILE, backup_file)
                except OSError:
                    pass  # Backup creation failed, continue anyway
            
            # Save new data
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Data saved successfully. New balance: {data['current_balance']:,.0f} VNĐ")
            
            # Async GitHub operations
            asyncio.create_task(self._async_github_operations(data))
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            # Restore backup if save failed
            if os.path.exists(f"{DATA_FILE}.backup"):
                try:
                    os.replace(f"{DATA_FILE}.backup", DATA_FILE)
                except OSError:
                    pass
    
    async def _async_github_operations(self, data: Dict[str, Any]) -> None:
        """Perform GitHub operations asynchronously."""
        try:
            # Push JSON file
            await asyncio.get_event_loop().run_in_executor(
                None, push_to_github, DATA_FILE, "Update expenses JSON via Discord bot"
            )
            
            # Convert and push CSV
            if convert_json_to_csv(data, "03.my_data.csv"):
                await asyncio.get_event_loop().run_in_executor(
                    None, push_to_github, "03.my_data.csv", "Update 03.my_data.csv from expenses.json"
                )
        except Exception as e:
            logger.error(f"GitHub operations failed: {e}")

# Initialize global data cache
data_cache = DataCache()

# --- Optimized Git operations ---
def push_to_github(file_path: str, commit_message: str) -> None:
    """
    Optimized GitHub push operation with better error handling and performance.
    """
    try:
        repo = git.Repo(LOCAL_REPO_PATH)
        repo.git.add(file_path)
        
        # Check for actual changes more efficiently
        status_output = repo.git.status('--porcelain=v1', file_path)
        
        if not status_output.strip().startswith(('M ', 'A ', 'MM', '?? ')):
            logger.info(f"No changes to {file_path} detected, skipping commit and push.")
            return
        
        repo.index.commit(commit_message)
        
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            raise ValueError("GITHUB_TOKEN environment variable not set.")
        
        # Optimize remote URL construction
        remote_url = f"https://oauth2:{github_token}@github.com/itsngocanh/budget-tracking.git"
        
        origin = repo.remote(name='origin')
        origin.set_url(remote_url)
        
        # Use more efficient push
        origin.push(refspec=f'{GIT_BRANCH}:{GIT_BRANCH}', set_upstream=True)
        logger.info(f"Successfully pushed {file_path} to GitHub on branch {GIT_BRANCH}")
        
    except git.InvalidGitRepositoryError:
        logger.error(f"'{LOCAL_REPO_PATH}' is not a valid Git repository.")
    except Exception as e:
        logger.error(f"Git operation failed for {file_path}: {e}")

# --- Optimized CSV conversion ---
def convert_json_to_csv(json_data: Dict[str, Any], csv_filename: str = "data/03.my_data.csv") -> bool:
    """
    Optimized JSON to CSV conversion with better performance.
    """
    try:
        header = ["Type", "Amount", "Subtype", "Description", "Timestamp"]
        rows = []
        
        # Pre-allocate lists for better performance
        income_rows = [(item["amount"], item["subtype"], item["description"], item["timestamp"]) 
                      for item in json_data["income"]]
        outcome_rows = [(-item["amount"], item["subtype"], item["description"], item["timestamp"]) 
                       for item in json_data["outcome"]]
        saving_rows = [(item["amount"], item["subtype"], item["description"], item["timestamp"]) 
                      for item in json_data["savings"]]
        
        # Combine all rows efficiently
        rows.extend([["income"] + list(row) for row in income_rows])
        rows.extend([["outcome"] + list(row) for row in outcome_rows])
        rows.extend([["saving"] + list(row) for row in saving_rows])
        
        # Sort by timestamp efficiently
        rows.sort(key=lambda x: x[4])
        
        # Write to CSV with optimized settings
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            writer.writerows(rows)
        
        logger.info(f"Successfully converted expenses.json to {csv_filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting JSON to CSV: {e}")
        return False

# --- Optimized daily tasks ---
@tasks.loop(time=time(2, 0, 0, tzinfo=timezone.utc))
async def daily_balance_check():
    """Optimized daily balance check task."""
    await bot.wait_until_ready()
    
    try:
        channel = bot.get_channel(DAILY_MESSAGE_CHANNEL_ID)
        if not channel:
            logger.error(f"Channel with ID {DAILY_MESSAGE_CHANNEL_ID} not found.")
            return
        
        current_data = data_cache.get_data()
        current_balance = current_data.get('current_balance', 0.0)
        
        message = (
            f"Cập nhật số dư hàng ngày ({datetime.now().strftime('%d-%m-%Y')}):\n"
            f"Số dư tiền mặt hiện tại của bạn là: **{current_balance:,.0f} VNĐ**"
        )
        
        if current_balance < 0:
            message += "\n⚠️ **Cảnh báo: Số dư của bạn đang âm! Vui lòng chú ý chi tiêu để tránh nợ nần.**"
        
        await channel.send(message)
        logger.info(f"Daily balance message sent to channel {channel.id}")
        
    except discord.Forbidden:
        logger.error(f"Bot does not have permission to send messages in channel {DAILY_MESSAGE_CHANNEL_ID}.")
    except Exception as e:
        logger.error(f"Error sending daily balance message: {e}")

@tasks.loop(time=CSV_EXPORT_TIME_UTC)
async def daily_csv_export():
    """Optimized daily CSV export task."""
    logger.info("Attempting daily CSV export...")
    
    try:
        channel = bot.get_channel(DAILY_CSV_EXPORT_CHANNEL_ID)
        if not channel:
            logger.error(f"Daily CSV export channel with ID {DAILY_CSV_EXPORT_CHANNEL_ID} not found.")
            return
        
        csv_filename = "03.my_data_daily_export.csv"
        current_data = data_cache.get_data()
        
        if convert_json_to_csv(current_data, csv_filename):
            try:
                await channel.send(
                    f"📊 Báo cáo dữ liệu hàng ngày tính đến {datetime.now().strftime('%d/%m/%Y %H:%M')}:",
                    file=discord.File(csv_filename)
                )
                logger.info(f"Daily CSV exported and sent to channel {channel.name}.")
                
                # Clean up file
                try:
                    os.remove(csv_filename)
                except OSError:
                    pass
                    
            except Exception as e:
                logger.error(f"Error sending daily CSV: {e}")
        else:
            logger.error("Failed to convert data to CSV for daily export.")
            
    except Exception as e:
        logger.error(f"Daily CSV export failed: {e}")

# --- Bot event handlers ---
@bot.event
async def on_ready():
    """Optimized bot ready event handler."""
    logger.info(f'{bot.user} has connected to Discord!')
    
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} command(s)")
    except Exception as e:
        logger.error(f"Error syncing commands: {e}")
    
    # Start background tasks
    daily_balance_check.start()
    daily_csv_export.start()
    
    logger.info("Bot is ready and all tasks are running!")

# --- Optimized command handlers ---
@bot.tree.command(name="thu", description="Ghi nhận một khoản thu nhập mới.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    amount="Số tiền thu nhập (ví dụ: 100000)",
    subtype="Loại thu nhập (ví dụ: Lương, Thưởng)",
    description="Mô tả chi tiết khoản thu nhập (tùy chọn)"
)
async def income(interaction: discord.Interaction, amount: float, subtype: str, description: str = None):
    """Optimized income command handler."""
    await interaction.response.defer(ephemeral=False)
    
    try:
        current_data = data_cache.get_data()
        
        # Create income entry
        income_entry = {
            "amount": amount,
            "subtype": subtype,
            "description": description or "Thu nhập",
            "timestamp": datetime.now().isoformat()
        }
        
        # Update data
        current_data["income"].append(income_entry)
        current_data["current_balance"] += amount
        
        # Save data (this will trigger async GitHub operations)
        data_cache.update_data(current_data)
        
        await interaction.edit_original_response(
            content=f"✅ **Thu nhập đã được ghi nhận:**\n"
            f"**Số tiền:** {amount:,.0f} VNĐ\n"
            f"**Loại:** {subtype}\n"
            f"**Mô tả:** {description or 'Thu nhập'}\n"
            f"**Số dư hiện tại:** {current_data['current_balance']:,.0f} VNĐ"
        )
        
        logger.info(f"Income command executed by {interaction.user.name}. Amount: {amount}")
        
    except Exception as e:
        logger.error(f"Error in income command: {e}")
        await interaction.edit_original_response(content=f"Có lỗi xảy ra: {e}")

@bot.tree.command(name="chi", description="Ghi nhận một khoản chi tiêu mới.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    amount="Số tiền chi tiêu (ví dụ: 50000)",
    subtype="Loại chi tiêu (ví dụ: Ăn uống, Di chuyển)",
    description="Mô tả chi tiết khoản chi tiêu (tùy chọn)"
)
async def outcome(interaction: discord.Interaction, amount: float, subtype: str, description: str = None):
    """Optimized outcome command handler."""
    await interaction.response.defer(ephemeral=False)
    
    try:
        current_data = data_cache.get_data()
        
        # Create outcome entry
        outcome_entry = {
            "amount": amount,
            "subtype": subtype,
            "description": description or "Chi tiêu",
            "timestamp": datetime.now().isoformat()
        }
        
        # Update data
        current_data["outcome"].append(outcome_entry)
        current_data["current_balance"] -= amount
        
        # Save data
        data_cache.update_data(current_data)
        
        await interaction.edit_original_response(
            content=f"💸 **Chi tiêu đã được ghi nhận:**\n"
            f"**Số tiền:** {amount:,.0f} VNĐ\n"
            f"**Loại:** {subtype}\n"
            f"**Mô tả:** {description or 'Chi tiêu'}\n"
            f"**Số dư hiện tại:** {current_data['current_balance']:,.0f} VNĐ"
        )
        
        logger.info(f"Outcome command executed by {interaction.user.name}. Amount: {amount}")
        
    except Exception as e:
        logger.error(f"Error in outcome command: {e}")
        await interaction.edit_original_response(content=f"Có lỗi xảy ra: {e}")

@bot.tree.command(name="tietkiem", description="Ghi nhận một khoản tiền tiết kiệm mới và trừ vào số dư.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    amount="Số tiền tiết kiệm (ví dụ: 100000)",
    subtype="Loại tiết kiệm (ví dụ: Tiết kiệm dài hạn, Quỹ khẩn cấp)",
    description="Mô tả chi tiết khoản tiết kiệm (tùy chọn)"
)
async def saving(interaction: discord.Interaction, amount: float, subtype: str, description: str = None):
    """Optimized saving command handler."""
    await interaction.response.defer(ephemeral=False)
    
    try:
        current_data = data_cache.get_data()
        
        # Create saving entry
        saving_entry = {
            "amount": amount,
            "subtype": subtype,
            "description": description or "Tiết kiệm",
            "timestamp": datetime.now().isoformat()
        }
        
        # Update data
        current_data["savings"].append(saving_entry)
        current_data["current_balance"] -= amount
        
        # Save data
        data_cache.update_data(current_data)
        
        await interaction.edit_original_response(
            content=f"🏦 **Tiết kiệm đã được ghi nhận:**\n"
            f"**Số tiền:** {amount:,.0f} VNĐ\n"
            f"**Loại:** {subtype}\n"
            f"**Mô tả:** {description or 'Tiết kiệm'}\n"
            f"**Số dư hiện tại:** {current_data['current_balance']:,.0f} VNĐ"
        )
        
        logger.info(f"Saving command executed by {interaction.user.name}. Amount: {amount}")
        
    except Exception as e:
        logger.error(f"Error in saving command: {e}")
        await interaction.edit_original_response(content=f"Có lỗi xảy ra: {e}")

@bot.tree.command(name="balance", description="Hiển thị số dư tiền mặt hiện tại của bạn.", guild=discord.Object(id=GUILD_ID))
async def balance(interaction: discord.Interaction):
    """Optimized balance command handler."""
    try:
        current_data = data_cache.get_data()
        current_balance = current_data.get('current_balance', 0.0)
        
        await interaction.response.send_message(
            f"💰 **Số dư tiền mặt hiện tại:** {current_balance:,.0f} VNĐ",
            ephemeral=True
        )
        
        logger.info(f"Balance command executed by {interaction.user.name}")
        
    except Exception as e:
        logger.error(f"Error in balance command: {e}")
        await interaction.response.send_message(f"Có lỗi xảy ra: {e}", ephemeral=True)

@bot.tree.command(name="add", description="Thêm một khoản tiền vào số dư của bạn.", guild=discord.Object(id=GUILD_ID))
async def add(interaction: discord.Interaction, amount: float):
    """Optimized add command handler."""
    await interaction.response.defer(ephemeral=False)
    
    try:
        current_data = data_cache.get_data()
        current_data["current_balance"] += amount
        data_cache.update_data(current_data)
        
        await interaction.edit_original_response(
            content=f"✅ **Đã thêm {amount:,.0f} VNĐ vào số dư.**\n"
            f"**Số dư mới:** {current_data['current_balance']:,.0f} VNĐ"
        )
        
        logger.info(f"Add command executed by {interaction.user.name}. Amount: {amount}")
        
    except Exception as e:
        logger.error(f"Error in add command: {e}")
        await interaction.edit_original_response(content=f"Có lỗi xảy ra: {e}")

@bot.tree.command(name="subtract", description="Trừ một khoản tiền khỏi số dư của bạn.", guild=discord.Object(id=GUILD_ID))
async def subtract(interaction: discord.Interaction, amount: float):
    """Optimized subtract command handler."""
    await interaction.response.defer(ephemeral=False)
    
    try:
        current_data = data_cache.get_data()
        current_data["current_balance"] -= amount
        data_cache.update_data(current_data)
        
        await interaction.edit_original_response(
            content=f"💸 **Đã trừ {amount:,.0f} VNĐ khỏi số dư.**\n"
            f"**Số dư mới:** {current_data['current_balance']:,.0f} VNĐ"
        )
        
        logger.info(f"Subtract command executed by {interaction.user.name}. Amount: {amount}")
        
    except Exception as e:
        logger.error(f"Error in subtract command: {e}")
        await interaction.edit_original_response(content=f"Có lỗi xảy ra: {e}")

# --- Additional optimized commands ---
@bot.tree.command(name="export_csv", description="Xuất dữ liệu thu chi và tiết kiệm ra file CSV.", guild=discord.Object(id=GUILD_ID))
async def export_csv(interaction: discord.Interaction):
    """Optimized CSV export command."""
    await interaction.response.defer(ephemeral=False)
    
    try:
        current_data = data_cache.get_data()
        csv_filename = "03.my_data.csv"
        
        if convert_json_to_csv(current_data, csv_filename):
            try:
                await interaction.edit_original_response(
                    content="✅ **Dữ liệu đã được xuất ra file CSV thành công!**\n"
                    "File đã được đồng bộ với GitHub repository."
                )
                # Send the file as a Discord attachment
                await interaction.followup.send(
                    content="📊 **Đây là file CSV của bạn:**",
                    file=discord.File(csv_filename)
                )
                
                # Clean up the local file after sending
                try:
                    os.remove(csv_filename)
                except OSError:
                    pass  # File cleanup failed, but that's okay
                    
            except Exception as e:
                logger.error(f"Error sending CSV file: {e}")
                await interaction.edit_original_response(
                    content="✅ **CSV đã được tạo và đồng bộ với GitHub, nhưng có lỗi khi gửi file qua Discord.**"
                )
        else:
            await interaction.edit_original_response(content="❌ **Có lỗi xảy ra khi xuất file CSV.**")
            
        logger.info(f"Export CSV command executed by {interaction.user.name}")
        
    except Exception as e:
        logger.error(f"Error in export_csv command: {e}")
        await interaction.edit_original_response(content=f"Có lỗi xảy ra: {e}")

@bot.tree.command(name="total", description="Hiển thị tổng thu nhập, chi tiêu và tiết kiệm ban đầu.", guild=discord.Object(id=GUILD_ID))
async def total(interaction: discord.Interaction):
    """Optimized total command handler."""
    try:
        current_data = data_cache.get_data()
        
        embed = discord.Embed(
            title="📊 Tổng quan tài chính",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="💰 Tổng thu nhập ban đầu",
            value=f"{current_data['initial_income_total']:,.0f} VNĐ",
            inline=True
        )
        embed.add_field(
            name="💸 Tổng chi tiêu ban đầu",
            value=f"{current_data['initial_outcome_total']:,.0f} VNĐ",
            inline=True
        )
        embed.add_field(
            name="🏦 Tổng tiết kiệm ban đầu",
            value=f"{current_data['initial_savings_total']:,.0f} VNĐ",
            inline=True
        )
        embed.add_field(
            name="💳 Số dư hiện tại",
            value=f"{current_data['current_balance']:,.0f} VNĐ",
            inline=False
        )
        
        await interaction.response.send_message(embed=embed, ephemeral=True)
        logger.info(f"Total command executed by {interaction.user.name}")
        
    except Exception as e:
        logger.error(f"Error in total command: {e}")
        await interaction.response.send_message(f"Có lỗi xảy ra: {e}", ephemeral=True)

# --- Entry point ---
if __name__ == '__main__':
    try:
        logger.info("Starting Discord bot...")
        bot.run(TOKEN)
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        exit(1)