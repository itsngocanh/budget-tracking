import discord
from discord.ext import commands, tasks
import json
import os
from datetime import datetime, time, timezone
import csv # Import csv module
import git # Import the git module
from dotenv import load_dotenv # type: ignore # For loading environment variables from a .env file

# Load environment variables from .env file
load_dotenv()

# Define intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# Initialize the bot with intents
bot = commands.Bot(command_prefix="!", intents=intents)

# --- Configuration ---
# Get Discord Token from environment variable
TOKEN = os.getenv('DISCORD_TOKEN') 
if not TOKEN:
    print("Error: DISCORD_TOKEN environment variable not set. Please set it in your .env file.")
    exit()

# Get Guild ID from environment variable, with a default if not found
GUILD_ID = int(os.getenv('GUILD_ID', 1378927251877531799)) 
DATA_FILE = 'expenses.json'

# --- NEW CONFIGURATION FOR DAILY MESSAGE ---
DAILY_MESSAGE_CHANNEL_ID = int(os.getenv('DAILY_MESSAGE_CHANNEL_ID', 1379091155538546769)) 

# --- NEW CONFIGURATION FOR DAILY CSV EXPORT ---
DAILY_CSV_EXPORT_CHANNEL_ID = int(os.getenv('DAILY_CSV_EXPORT_CHANNEL_ID', 1385852449063174154)) 
# Example: 3:00 AM UTC = 10:00 AM ICT (Vietnam time)
CSV_EXPORT_TIME_UTC = time(3, 0, 0, tzinfo=timezone.utc)

# --- NEW CONFIGURATION FOR GITHUB AUTOMATION ---
# Ensure your local repository path is correct. This is where your .git folder resides.
LOCAL_REPO_PATH = '.' # Assuming your script is run from the root of your Git repository
GIT_BRANCH = 'main' # Or 'master', depending on your default branch name

# --- Data Management ---
def load_expenses():
    print("Attempting to load expenses data...") # Debug print
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # Ensure all top-level keys exist and initialize new ones if missing
                data.setdefault("income", [])
                data.setdefault("outcome", [])
                data.setdefault("savings", [])
                data.setdefault("current_balance", 0.0)
                # NEW: Initialize initial category totals if missing
                data.setdefault("initial_income_total", 0.0)
                data.setdefault("initial_outcome_total", 0.0)
                data.setdefault("initial_savings_total", 0.0)
                print(f"Expenses data loaded successfully. Current balance: {data['current_balance']}") # Debug print
                return data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {DATA_FILE}: {e}") # Debug print for JSON error
                # If file is corrupted or empty, start fresh but notify
                return {
                    "income": [],
                    "outcome": [],
                    "savings": [],
                    "current_balance": 0.0,
                    "initial_income_total": 0.0,
                    "initial_outcome_total": 0.0,
                    "initial_savings_total": 0.0
                }
    print("No expenses data file found or new file created. Initializing new data structure.") # Debug print
    return {
        "income": [],
        "outcome": [],
        "savings": [],
        "current_balance": 0.0,
        "initial_income_total": 0.0,
        "initial_outcome_total": 0.0,
        "initial_savings_total": 0.0
    }
    pass

def save_expenses(data):
    """Saves expense data to a JSON file and triggers Git push for both JSON and converted CSV.
    This function acts as the central point for data persistence and GitHub synchronization.
    """
    print("Attempting to save expenses data...")
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Expenses data saved successfully. New balance: {data['current_balance']:,.0f} VNĐ")

        # Try to push the JSON file
        try:
            push_to_github(DATA_FILE, "Update expenses JSON via Discord bot")
        except Exception as e:
            print(f"Failed to push {DATA_FILE} to GitHub: {e}")

        # Convert to CSV and push the CSV as well for the Streamlit dashboard
        # This ensures the dashboard always has the latest data.
        if convert_json_to_csv(data, "03.my_data.csv"):
            try:
                push_to_github("03.my_data.csv", "Update 03.my_data.csv from expenses.json")
            except Exception as e:
                print(f"Failed to push 03.my_data.csv to GitHub: {e}")
    except Exception as e:
        print(f"Error saving expenses data or during Git operation: {e}")

        # --- NEW FUNCTION: Git Automation ---
def push_to_github(file_path: str, commit_message: str):
    """
    Adds, commits, and pushes a specified file to the GitHub repository.
    Authenticates using a GitHub Personal Access Token loaded from GITHUB_TOKEN environment variable.
    Only performs commit/push if actual changes are detected for the file.
    """
    try:
        repo = git.Repo(LOCAL_REPO_PATH)
        repo.git.add(file_path) # Stage the file for commit
        
        # Check if there are any *actual* changes to commit for the specified file
        # Use git status --porcelain=v1 to check file status (M: modified, A: added, ??: untracked)
        status_output = repo.git.status('--porcelain=v1', file_path)

        file_changed = False
        if status_output:
            # A file is considered changed if its status indicates modification (' M', 'MM'),
            # staging ('A '), or being untracked ('?? ').
            # Since we call repo.git.add(file_path) first, we primarily expect 'A ' or 'M '.
            if status_output.strip().startswith(('M ', 'A ', 'MM', '?? ')):
                file_changed = True

        if file_changed:
            repo.index.commit(commit_message)

            github_token = os.getenv('GITHUB_TOKEN')
            if not github_token:
                raise ValueError("GITHUB_TOKEN environment variable not set. Cannot push to GitHub. Please set it in your .env file.")

            # Construct the remote URL with the token for authentication
            # IMPORTANT: Replace 'your-github-username' and 'your-repo-name' with your actual values!
            remote_url = f"https://oauth2:{github_token}@github.com/itsngocanh/budget-tracking.git"

             # Get the 'origin' remote (standard name for main remote) and update its URL
            # THIS IS THE FIXED LINE: Use set_url() to correctly change the remote's URL
            origin = repo.remote(name='origin')
            origin.set_url(remote_url)

            # Push the changes to the specified branch
            origin.push(refspec=f'{GIT_BRANCH}:{GIT_BRANCH}')
            print(f"Successfully pushed {file_path} to GitHub on branch {GIT_BRANCH}")
        else:
            print(f"No changes to {file_path} detected, skipping commit and push.")

    except git.InvalidGitRepositoryError:
        print(f"Error: '{LOCAL_REPO_PATH}' is not a valid Git repository. Please ensure it's initialized and the path is correct.")
        print("Hint: Run 'git init' in your project folder, then 'git remote add origin YOUR_REPO_URL' if you haven't.")
    except Exception as e:
        print(f"An error occurred during Git operation for {file_path}: {e}")

# --- Function to convert JSON to CSV (for Streamlit) ---
def convert_json_to_csv(json_data, csv_filename="03.my_data.csv"):
    """
    Converts the expense JSON data into a CSV format suitable for the Streamlit dashboard.
    It processes income, outcome, and savings entries, sorting them by timestamp.
    """
    header = ["Type", "Amount", "Subtype", "Description", "Timestamp"]
    rows = []

    # Add income entries
    for item in json_data["income"]:
        rows.append(["income", item["amount"], item["subtype"], item["description"], item["timestamp"]])

    # Add outcome entries (amount as negative for consistency in dashboard calculations)
    for item in json_data["outcome"]:
        rows.append(["outcome", -item["amount"], item["subtype"], item["description"], item["timestamp"]])

    # Add savings entries
    for item in json_data["savings"]:
        rows.append(["saving", item["amount"], item["subtype"], item["description"], item["timestamp"]])

    # Sort all entries by timestamp to maintain chronological order in CSV
    rows.sort(key=lambda x: x[4])

    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Successfully converted expenses.json to {csv_filename}")
        return True
    except Exception as e:
        print(f"Error converting JSON to CSV: {e}")
        return False

expenses_data = load_expenses()

# --- Daily Balance Check Task ---
@tasks.loop(time=time(2, 0, 0, tzinfo=timezone.utc)) # 2:00 AM UTC = 9:00 AM ICT (Vietnam time)
async def daily_balance_check():
    await bot.wait_until_ready()
    channel = bot.get_channel(DAILY_MESSAGE_CHANNEL_ID)

    if channel:
        current_expenses_data = load_expenses()
        current_balance = current_expenses_data.get('current_balance', 0.0)

        message = (
            f"Cập nhật số dư hàng ngày ({datetime.now().strftime('%d-%m-%Y')}):\n"
            f"Số dư tiền mặt hiện tại của bạn là: **{current_balance:,.0f} VNĐ**"
        )
        if current_balance < 0:
            message += "\n⚠️ **Cảnh báo: Số dư của bạn đang âm! Vui lòng chú ý chi tiêu để tránh nợ nần.**"

        try:
            await channel.send(message)
            print(f"Daily balance message sent to channel {channel.id}")
        except discord.Forbidden:
            print(f"Bot does not have permission to send messages in channel {channel.id}.")
        except Exception as e:
            print(f"Error sending daily balance message: {e}")
    else:
        print(f"Channel with ID {DAILY_MESSAGE_CHANNEL_ID} not found or bot does not have access.")

# --- NEW DAILY CSV EXPORT TASK ---
# ...
@tasks.loop(time=CSV_EXPORT_TIME_UTC) # Task runs daily at the specified UTC time
async def daily_csv_export():
    """Automatically exports the current data to a CSV file and sends it to a designated channel."""
    print("Attempting daily CSV export...")
    channel = bot.get_channel(DAILY_CSV_EXPORT_CHANNEL_ID)
    if not channel:
        print(f"Error: Daily CSV export channel with ID {DAILY_CSV_EXPORT_CHANNEL_ID} not found. Please check configuration.")
        return

    # A separate filename for daily exports to avoid conflicts with the Streamlit data file if needed
    csv_filename = "03.my_data_daily_export.csv" 

    # Ensure the latest data is used by reloading before conversion
    global expenses_data
    expenses_data = load_expenses()

    if convert_json_to_csv(expenses_data, csv_filename):
        try:
            await channel.send(f"📊 Báo cáo dữ liệu hàng ngày tính đến {datetime.now().strftime('%d/%m/%Y %H:%M')}:",
                               file=discord.File(csv_filename))
            print(f"Daily CSV exported and sent to channel {channel.name}.")
            os.remove(csv_filename) # Clean up the local file after sending to avoid clutter
        except Exception as e:
            print(f"Failed to send daily CSV export: {e}")
    else:
        print("Failed to convert JSON to CSV for daily export. Check console for details.")
# ...

# --- Bot Events ---
@bot.event
async def on_ready():
    """Event handler for when the bot successfully connects to Discord.
    It syncs slash commands and starts the daily scheduled tasks.
    """
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    print('------')
    # Sync slash commands to make them available in the Discord guild
    try:
        synced = await bot.tree.sync(guild=discord.Object(id=GUILD_ID))
        print(f"Synced {len(synced)} commands for guild {GUILD_ID}")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

    if not daily_balance_check.is_running():
        daily_balance_check.start()
        print("Daily balance check task started.")

    # Start the new daily CSV export task
    if not daily_csv_export.is_running():
        daily_csv_export.start()
        print("Daily CSV export task started.")


# --- Commands ---

@bot.tree.command(name="thu", description="Ghi nhận một khoản thu nhập mới.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    amount="Số tiền thu nhập (ví dụ: 100000)",
    subtype="Loại thu nhập (ví dụ: Lương, Thưởng)",
    description="Mô tả chi tiết khoản thu nhập (tùy chọn)"
)
async def income(interaction: discord.Interaction, amount: float, subtype: str, description: str = None):
    try:
        if amount <= 0:
            await interaction.response.send_message("Số tiền thu nhập phải lớn hơn 0.")
            return

        print(f"Before income: current_balance = {expenses_data['current_balance']}") # Debug print
        expenses_data["income"].append({
            "amount": amount,
            "subtype": subtype,
            "description": description if description else "Không có mô tả",
            "timestamp": datetime.now().isoformat()
        })
        expenses_data["current_balance"] += amount
        print(f"After income: current_balance = {expenses_data['current_balance']}") # Debug print
        save_expenses(expenses_data)
        await interaction.response.send_message(
            f"Đã ghi nhận khoản thu: **{amount:,.0f} VNĐ** (Loại: {subtype}, Mô tả: {description if description else 'Không có mô tả'})\n"
            f"Số dư hiện tại: **{expenses_data['current_balance']:,.0f} VNĐ**"
        )
    except Exception as e:
        await interaction.response.send_message(f"Có lỗi xảy ra khi ghi nhận thu nhập: {e}")
        print(f"Error in /thu command: {e}") # Debug print for command error

@bot.tree.command(name="chi", description="Ghi nhận một khoản chi tiêu mới.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    amount="Số tiền chi tiêu (ví dụ: 50000)",
    subtype="Loại chi tiêu (ví dụ: Ăn uống, Di chuyển)",
    description="Mô tả chi tiết khoản chi tiêu (tùy chọn)"
)
async def outcome(interaction: discord.Interaction, amount: float, subtype: str, description: str = None):
    try:
        if amount <= 0:
            await interaction.response.send_message("Số tiền chi tiêu phải lớn hơn 0.")
            return

        print(f"Before outcome: current_balance = {expenses_data['current_balance']}") # Debug print
        expenses_data["outcome"].append({
            "amount": amount,
            "subtype": subtype,
            "description": description if description else "Không có mô tả",
            "timestamp": datetime.now().isoformat()
        })
        expenses_data["current_balance"] -= amount
        print(f"After outcome: current_balance = {expenses_data['current_balance']}") # Debug print

        save_expenses(expenses_data)

        response_message = (
            f"Đã ghi nhận khoản chi: **{amount:,.0f} VNĐ** (Loại: {subtype}, Mô tả: {description if description else 'Không có mô tả'})\n"
            f"Số dư hiện tại: **{expenses_data['current_balance']:,.0f} VNĐ**"
        )
        if expenses_data["current_balance"] < 0:
            response_message += "\n⚠️ **Cảnh báo: Chi tiêu này đã khiến số dư của bạn âm! Vui lòng chú ý quản lý tài chính.**"

        await interaction.response.send_message(response_message)
    except Exception as e:
        await interaction.response.send_message(f"Có lỗi xảy ra khi ghi nhận chi tiêu: {e}")
        print(f"Error in /chi command: {e}") # Debug print for command error

@bot.tree.command(name="tietkiem", description="Ghi nhận một khoản tiền tiết kiệm mới và trừ vào số dư.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    amount="Số tiền tiết kiệm (ví dụ: 100000)",
    subtype="Loại tiết kiệm (ví dụ: Tiết kiệm dài hạn, Quỹ khẩn cấp)",
    description="Mô tả chi tiết khoản tiết kiệm (tùy chọn)"
)
async def saving(interaction: discord.Interaction, amount: float, subtype: str, description: str = None):
    try:
        if amount <= 0:
            await interaction.response.send_message("Số tiền tiết kiệm phải lớn hơn 0.")
            return

        print(f"Before saving: current_balance = {expenses_data['current_balance']}") # Debug print
        expenses_data["savings"].append({
            "amount": amount,
            "subtype": subtype,
            "description": description if description else "Không có mô tả",
            "timestamp": datetime.now().isoformat()
        })
        expenses_data["current_balance"] -= amount
        print(f"After saving: current_balance = {expenses_data['current_balance']}") # Debug print

        save_expenses(expenses_data)

        response_message = (
            f"Đã chuyển **{amount:,.0f} VNĐ** vào tiết kiệm (Loại: {subtype}, Mô tả: {description if description else 'Không có mô tả'})\n"
            f"Số dư hiện tại: **{expenses_data['current_balance']:,.0f} VNĐ**"
        )
        if expenses_data["current_balance"] < 0:
            response_message += "\n⚠️ **Cảnh báo: Việc tiết kiệm này đã khiến số dư của bạn âm! Vui lòng cân nhắc kế hoạch tài chính.**"

        await interaction.response.send_message(response_message)
    except Exception as e:
        await interaction.response.send_message(f"Có lỗi xảy ra khi ghi nhận tiết kiệm: {e}")
        print(f"Error in /tietkiem command: {e}") # Debug print for command error


@bot.tree.command(name="tong_thu", description="Tính tổng thu nhập.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    from_date="Ngày bắt đầu (YYYY-MM-DD, tùy chọn)",
    to_date="Ngày kết thúc (YYYY-MM-DD, tùy chọn)",
    subtype="Loại thu nhập cụ thể (tùy chọn)"
)
async def sum_income(interaction: discord.Interaction, from_date: str = None, to_date: str = None, subtype: str = None):
    # Load the latest data to ensure up-to-date calculations
    current_expenses_data = load_expenses()

    total = current_expenses_data.get("initial_income_total", 0.0) # Include initial total
    filtered_items = []

    start_dt = None
    end_dt = None

    if from_date:
        try:
            start_dt = datetime.strptime(from_date, "%Y-%m-%d")
        except ValueError:
            await interaction.response.send_message("Định dạng ngày bắt đầu không hợp lệ. Vui lòng sử dụng `%Y-%m-%d`.")
            return
    if to_date:
        try:
            end_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        except ValueError:
            await interaction.response.send_message("Định dạng ngày kết thúc không hợp lệ. Vui lòng sử dụng `%Y-%m-%d`.")
            return

    for item in current_expenses_data["income"]: # Use current_expenses_data
        item_dt = datetime.fromisoformat(item["timestamp"])

        if start_dt and item_dt < start_dt:
            continue
        if end_dt and item_dt > end_dt:
            continue

        if subtype and subtype.lower() not in item["subtype"].lower():
            continue

        filtered_items.append(item)
        total += item["amount"]

    if not filtered_items and current_expenses_data.get("initial_income_total", 0.0) == 0:
        await interaction.response.send_message("Không tìm thấy khoản thu nào phù hợp với tiêu chí của bạn.")
        return

    response_msg = f"Tổng thu nhập: **{total:,.0f} VNĐ**\n"
    if current_expenses_data.get("initial_income_total", 0.0) > 0:
        response_msg += f"(Bao gồm {current_expenses_data['initial_income_total']:,.0f} VNĐ từ mục nhập ban đầu)\n"

    if filtered_items:
        # Define a local limit for display here
        display_limit_sum_commands = 10
        response_msg += "Chi tiết thu nhập giao dịch gần đây:\n"
        for item in filtered_items[-min(display_limit_sum_commands, len(filtered_items)):]:
            response_msg += f"- {datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d')}: {item['amount']:,.0f} VNĐ ({item['subtype']} - {item['description']})\n"
        if len(filtered_items) > display_limit_sum_commands:
            response_msg += f"... và còn {len(filtered_items) - display_limit_sum_commands} khoản thu khác.\n"

    await interaction.response.send_message(response_msg)

@bot.tree.command(name="tong_chi", description="Tính tổng chi tiêu.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    from_date="Ngày bắt đầu (YYYY-MM-%d, tùy chọn)",
    to_date="Ngày kết thúc (YYYY-MM-%d, tùy chọn)",
    subtype="Loại chi tiêu cụ thể (tùy chọn)"
)
async def sum_outcome(interaction: discord.Interaction, from_date: str = None, to_date: str = None, subtype: str = None):
    # Load the latest data
    current_expenses_data = load_expenses()

    total = current_expenses_data.get("initial_outcome_total", 0.0) # Include initial total
    filtered_items = []

    start_dt = None
    end_dt = None

    if from_date:
        try:
            start_dt = datetime.strptime(from_date, "%Y-%m-%d")
        except ValueError:
            await interaction.response.send_message("Định dạng ngày bắt đầu không hợp lệ. Vui lòng sử dụng `%Y-%m-%d`.")
            return
    if to_date:
        try:
            end_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        except ValueError:
            await interaction.response.send_message("Định dạng ngày kết thúc không hợp lệ. Vui lòng sử dụng `%Y-%m-%d`.")
            return

    for item in current_expenses_data["outcome"]: # Use current_expenses_data
        item_dt = datetime.fromisoformat(item["timestamp"])

        if start_dt and item_dt < start_dt:
            continue
        if end_dt and item_dt > end_dt:
            continue

        if subtype and subtype.lower() not in item["subtype"].lower():
            continue

        filtered_items.append(item)
        total += item["amount"]

    if not filtered_items and current_expenses_data.get("initial_outcome_total", 0.0) == 0:
        await interaction.response.send_message("Không tìm thấy khoản chi nào phù hợp với tiêu chí của bạn.")
        return

    response_msg = f"Tổng chi tiêu: **{total:,.0f} VNĐ**\n"
    if current_expenses_data.get("initial_outcome_total", 0.0) > 0:
        response_msg += f"(Bao gồm {current_expenses_data['initial_outcome_total']:,.0f} VNĐ từ mục nhập ban đầu)\n"

    if filtered_items:
        # Define a local limit for display here
        display_limit_sum_commands = 10
        response_msg += "Chi tiết chi tiêu giao dịch gần đây:\n"
        for item in filtered_items[-min(display_limit_sum_commands, len(filtered_items)):]:
            response_msg += f"- {datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d')}: {item['amount']:,.0f} VNĐ ({item['subtype']} - {item['description']})\n"
        if len(filtered_items) > display_limit_sum_commands:
            response_msg += f"... và còn {len(filtered_items) - display_limit_sum_commands} khoản chi khác.\n"

    await interaction.response.send_message(response_msg)

@bot.tree.command(name="tong_tietkiem", description="Tính tổng tiền tiết kiệm.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    from_date="Ngày bắt đầu (YYYY-MM-%d, tùy chọn)",
    to_date="Ngày kết thúc (YYYY-MM-%d, tùy chọn)",
    subtype="Loại tiết kiệm cụ thể (tùy chọn)"
)
async def sum_saving(interaction: discord.Interaction, from_date: str = None, to_date: str = None, subtype: str = None):
    # Load the latest data
    current_expenses_data = load_expenses()

    total = current_expenses_data.get("initial_savings_total", 0.0) # Include initial total
    filtered_items = []

    start_dt = None
    end_dt = None

    if from_date:
        try:
            start_dt = datetime.strptime(from_date, "%Y-%m-%d")
        except ValueError:
            await interaction.response.send_message("Định dạng ngày bắt đầu không hợp lệ. Vui lòng sử dụng `%Y-%m-%d`.")
            return
    if to_date:
        try:
            end_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        except ValueError:
            await interaction.response.send_message("Định dạng ngày kết thúc không hợp lệ. Vui lòng sử dụng `%Y-%m-%d`.")
            return

    for item in current_expenses_data["savings"]: # Use current_expenses_data
        item_dt = datetime.fromisoformat(item["timestamp"])

        if start_dt and item_dt < start_dt:
            continue
        if end_dt and item_dt > end_dt:
            continue

        if subtype and subtype.lower() not in item["subtype"].lower():
            continue

        filtered_items.append(item)
        total += item["amount"]

    if not filtered_items and current_expenses_data.get("initial_savings_total", 0.0) == 0:
        await interaction.response.send_message("Không tìm thấy khoản tiết kiệm nào phù hợp với tiêu chí của bạn.")
        return

    response_msg = f"Tổng tiền tiết kiệm: **{total:,.0f} VNĐ**\n"
    if current_expenses_data.get("initial_savings_total", 0.0) > 0:
        response_msg += f"(Bao gồm {current_expenses_data['initial_savings_total']:,.0f} VNĐ từ mục nhập ban đầu)\n"

    if filtered_items:
        # Define a local limit for display here
        display_limit_sum_commands = 10
        response_msg += "Chi tiết khoản tiết kiệm giao dịch gần đây:\n"
        for item in filtered_items[-min(display_limit_sum_commands, len(filtered_items)):]:
            response_msg += f"- {datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d')}: {item['amount']:,.0f} VNĐ ({item['subtype']} - {item['description']})\n"
        if len(filtered_items) > display_limit_sum_commands:
            response_msg += f"... và còn {len(filtered_items) - display_limit_sum_commands} khoản tiết kiệm khác.\n"

    await interaction.response.send_message(response_msg)

@bot.tree.command(name="clear_expense", description="Chọn và xóa một khoản thu, chi, hoặc tiết kiệm cụ thể.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    expense_type="Loại khoản cần xóa (income/outcome/saving)",
    index="Chỉ số chính xác của khoản cần xóa (dùng lệnh '/list_expenses' để xem)"
)
async def clear_expense(interaction: discord.Interaction, expense_type: str, index: int):
    expense_type_lower = expense_type.lower()
    if expense_type_lower not in ["income", "outcome", "saving"]:
        await interaction.response.send_message("Loại khoản không hợp lệ. Vui lòng chọn 'income', 'outcome', hoặc 'saving'.")
        return

    # Load the latest data
    current_expenses_data = load_expenses()

    key_mapping = {
        "income": "income",
        "outcome": "outcome",
        "saving": "savings"
    }
    actual_key = key_mapping.get(expense_type_lower)
    if not actual_key:
        await interaction.response.send_message("Lỗi nội bộ: Không thể ánh xạ loại khoản chi phí.")
        return

    target_list = current_expenses_data[actual_key]

    # Check if the provided index is within the valid range of the actual list
    if not (0 <= index < len(target_list)): # Index is now 0-based
        await interaction.response.send_message("Chỉ số không hợp lệ hoặc khoản không tồn tại. Vui lòng kiểm tra lại bằng lệnh '/list_expenses' mới nhất.")
        return

    try:
        removed_item = target_list.pop(index) # Use the index directly

        if expense_type_lower == "income":
            current_expenses_data["current_balance"] -= removed_item["amount"]
        elif expense_type_lower in ["outcome", "saving"]:
            current_expenses_data["current_balance"] += removed_item["amount"]

        save_expenses(current_expenses_data)

        global expenses_data
        expenses_data = load_expenses()

        await interaction.response.send_message(
            f"Đã xóa khoản **{expense_type_lower}**:\n"
            f"**Số tiền:** {removed_item['amount']:,.0f} VNĐ\n"
            f"**Loại:** {removed_item['subtype']}\n"
            f"**Mô tả:** {removed_item['description']}\n"
            f"**Thời gian:** {datetime.fromisoformat(removed_item['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Số dư hiện tại: **{expenses_data['current_balance']:,.0f} VNĐ**"
        )
    except Exception as e:
        await interaction.response.send_message(f"Có lỗi xảy ra khi xóa khoản: {e}")
        print(f"Error in /clear_expense command: {e}")

@bot.tree.command(name="list_expenses", description="Liệt kê các khoản thu/chi/tiết kiệm gần đây để hỗ trợ xóa.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    expense_type="Loại khoản cần liệt kê (income/outcome/saving, tùy chọn, mặc định là tất cả)"
)
async def list_expenses(interaction: discord.Interaction, expense_type: str = None):
    global expenses_data
    current_expenses_data = load_expenses() # Ensure freshest data

    # Prepare parts of the response to be joined later
    response_parts = []
    has_content = False

    # Define a maximum length for each message chunk to avoid Discord's 2000 character limit.
    # Using a slightly lower value like 1800 to be safe and account for embed overhead.
    MAX_CHUNK_LENGTH = 1800 
    
    # Defer the response since we might send multiple messages
    await interaction.response.defer(ephemeral=True)

    # --- Income Section ---
    income_lines = []
    if expense_type is None or expense_type.lower() == "income":
        income_lines.append("**💰 Thu nhập (Index: [Chỉ số thực tế]):**")
        if current_expenses_data["income"]:
            has_content = True
            for i, item in enumerate(current_expenses_data["income"]):
                try:
                    formatted_time = datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d %H:%M')
                except ValueError:
                    formatted_time = item['timestamp'] # Fallback if timestamp format is unexpected
                income_lines.append(f"[{i}] {formatted_time}: {item['amount']:,.0f} VNĐ ({item['subtype']} - {item['description']})")
        else:
            income_lines.append("Không có khoản thu nào.")
        income_lines.append("\n") # Add a separator

    # --- Outcome Section ---
    outcome_lines = []
    if expense_type is None or expense_type.lower() == "outcome":
        outcome_lines.append("**💸 Chi tiêu (Index: [Chỉ số thực tế]):**")
        if current_expenses_data["outcome"]:
            has_content = True
            for i, item in enumerate(current_expenses_data["outcome"]):
                try:
                    formatted_time = datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d %H:%M')
                except ValueError:
                    formatted_time = item['timestamp']
                outcome_lines.append(f"[{i}] {formatted_time}: {item['amount']:,.0f} VNĐ ({item['subtype']} - {item['description']})")
        else:
            outcome_lines.append("Không có khoản chi nào.")
        outcome_lines.append("\n") # Add a separator

    # --- Saving Section ---
    saving_lines = []
    if expense_type is None or expense_type.lower() == "saving":
        saving_lines.append("**🏦 Tiết kiệm (Index: [Chỉ số thực tế]):**")
        if current_expenses_data["savings"]:
            has_content = True
            for i, item in enumerate(current_expenses_data["savings"]):
                try:
                    formatted_time = datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d %H:%M')
                except ValueError:
                    formatted_time = item['timestamp']
                saving_lines.append(f"[{i}] {formatted_time}: {item['amount']:,.0f} VNĐ ({item['subtype']} - {item['description']})")
        else:
            saving_lines.append("Không có khoản tiết kiệm nào.")
        saving_lines.append("\n") # Add a separator

    if not has_content:
        await interaction.edit_original_response(content="Hiện tại không có dữ liệu thu chi hay tiết kiệm nào.")
        return

    all_content_lines = income_lines + outcome_lines + saving_lines

    # --- Pagination Logic ---
    current_chunk = []
    current_chunk_length = 0
    message_count = 0

    for line in all_content_lines:
        # Check if adding the next line would exceed the chunk limit
        if current_chunk_length + len(line) + 1 > MAX_CHUNK_LENGTH and current_chunk: # +1 for newline
            # If so, send the current chunk
            message_count += 1
            embed = discord.Embed(
                title=f"Danh sách giao dịch (Phần {message_count})",
                description="\n".join(current_chunk),
                color=discord.Color.blue()
            )
            embed.set_footer(text="Sử dụng lệnh /delete để xóa giao dịch theo chỉ số.")
            if message_count == 1: # First message uses edit_original_response
                await interaction.edit_original_response(embed=embed)
            else: # Subsequent messages use follow-up
                await interaction.followup.send(embed=embed, ephemeral=True)
            
            # Start a new chunk
            current_chunk = [line]
            current_chunk_length = len(line)
        else:
            # Otherwise, add the line to the current chunk
            current_chunk.append(line)
            current_chunk_length += len(line) + 1 # +1 for newline

    # Send any remaining content in the last chunk
    if current_chunk:
        message_count += 1
        embed = discord.Embed(
            title=f"Danh sách giao dịch (Phần {message_count})",
            description="\n".join(current_chunk),
            color=discord.Color.blue()
        )
        embed.set_footer(text="Sử dụng lệnh /delete để xóa giao dịch theo chỉ số.")
        if message_count == 1: # If only one message was needed
            await interaction.edit_original_response(embed=embed)
        else: # For the last of multiple messages
            await interaction.followup.send(embed=embed, ephemeral=True)

    print(f"List expenses command executed by {interaction.user.name}. Type: {expense_type if expense_type else 'all'}.")

@bot.tree.command(name="export_csv", description="Xuất dữ liệu thu chi và tiết kiệm ra file CSV.", guild=discord.Object(id=GUILD_ID))
async def export_csv(interaction: discord.Interaction):
    csv_filename = "03.my_data.csv" # Use the specific name for Streamlit data

    # Load the latest data for immediate export command
    current_expenses_data = load_expenses()

    header = ["Type", "Amount", "Subtype", "Description", "Timestamp"]
    rows = []

    for item in current_expenses_data["income"]:
        rows.append(["Income", item["amount"], item["subtype"], item["description"], item["timestamp"]])

    for item in current_expenses_data["outcome"]:
        rows.append(["Outcome", item["amount"], item["subtype"], item["description"], item["timestamp"]])

    for item in current_expenses_data["savings"]:
        rows.append(["Saving", item["amount"], item["subtype"], item["description"], item["timestamp"]])

    rows.sort(key=lambda x: x[4])

    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)

        await interaction.response.send_message(file=discord.File(csv_filename))
        os.remove(csv_filename) # Consider if you want to remove it immediately or leave it for Streamlit
        # After exporting, push to GitHub
        try:
            push_to_github(csv_filename, "Automated CSV export from Discord bot")
        except Exception as e:
            print(f"Failed to push exported CSV to GitHub: {e}")
    except Exception as e:
        await interaction.response.send_message(f"Có lỗi xảy ra khi xuất CSV: {e}")
        print(f"Error in /export_csv command: {e}") # Debug print for command error

# IMPORTANT: Also consider when the '03.my_data.csv' is initially created or intended to be updated.
# If 'expenses.json' is the primary source, and '03.my_data.csv' is generated *from* it,
# you'll need a separate process to convert JSON to CSV and then push the CSV.
# For simplicity, I've shown pushing DATA_FILE (expenses.json) and also added a call
# in export_csv if that's the point where '03.my_data.csv' is fresh.

# For Streamlit's 03.my_data.csv, you might need a separate mechanism.
# Your Streamlit app's load_data() function currently handles missing CSV by creating sample data.
# If your Discord bot is the *only* source of data, then the 'expenses.json' file is key.
# You could modify load_data() in 01.Expense.py to read from 'expenses.json' directly,
# or have a scheduled task on your bot to convert 'expenses.json' to '03.my_data.csv'
# and then push '03.my_data.csv'.

# If '03.my_data.csv' is manually maintained, then the Git automation would be less relevant
# for automated updates from the bot. Assuming the bot is the source,
# the save_expenses() is the place to trigger the push of expenses.json.
# If you intend 03.my_data.csv to be the shared file, you need a function to convert json to csv
# and trigger its push.

# Let's add a simple conversion and push logic if you decide `expenses.json` should be converted to `03.my_data.csv`
# and *that* CSV pushed.

# --- Function to convert JSON to CSV (for Streamlit) ---
def convert_json_to_csv(json_data, csv_filename="03.my_data.csv"):
    header = ["Type", "Amount", "Subtype", "Description", "Timestamp"]
    rows = []

    for item in json_data["income"]:
        rows.append(["income", item["amount"], item["subtype"], item["description"], item["timestamp"]])

    for item in json_data["outcome"]:
        rows.append(["outcome", -item["amount"], item["subtype"], item["description"], item["timestamp"]]) # Amount as negative for outcome

    for item in json_data["savings"]:
        rows.append(["saving", item["amount"], item["subtype"], item["description"], item["timestamp"]])

    rows.sort(key=lambda x: x[4]) # Sort by timestamp

    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Successfully converted expenses.json to {csv_filename}")
        return True
    except Exception as e:
        print(f"Error converting JSON to CSV: {e}")
        return False

# Modify save_expenses to also convert and push the CSV
def save_expenses(data):
    print("Attempting to save expenses data...")
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Expenses data saved successfully. New balance: {data['current_balance']}")
    
    # Try to push the JSON file
    try:
        push_to_github(DATA_FILE, "Update expenses JSON via Discord bot")
    except Exception as e:
        print(f"Failed to push {DATA_FILE} to GitHub: {e}")

    # Now, convert to CSV and push the CSV as well for the Streamlit dashboard
    if convert_json_to_csv(data, "03.my_data.csv"):
        try:
            push_to_github("03.my_data.csv", "Update 03.my_data.csv from expenses.json")
        except Exception as e:
            print(f"Failed to push 03.my_data.csv to GitHub: {e}")

@bot.tree.command(name="balance", description="Kiểm tra số dư tiền mặt hiện có.", guild=discord.Object(id=GUILD_ID))
async def balance(interaction: discord.Interaction):
    # Load current data from file to ensure it's up-to-date
    current_expenses_data = load_expenses()
    current_balance = current_expenses_data.get('current_balance', 0.0)
    response_message = f"Số dư tiền mặt hiện tại của bạn là: **{current_balance:,.0f} VNĐ**"
    if current_balance < 0:
        response_message += "\n⚠️ **Cảnh báo: Số dư của bạn đang âm! Vui lòng chú ý chi tiêu để tránh nợ nần.**"
    await interaction.response.send_message(response_message)

@bot.tree.command(name="input", description="Set or adjust initial amounts for cash balance or specific categories.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    amount="The amount to input.",
    category="Optional: Specify a category to adjust its initial total (Income, Outcome, Saving). If not specified, adds to overall cash balance."
)
@discord.app_commands.choices(
    category=[
        discord.app_commands.Choice(name="Income", value="income"),
        discord.app_commands.Choice(name="Outcome", value="outcome"),
        discord.app_commands.Choice(name="Saving", value="saving")
    ]
)
async def input_amount(interaction: discord.Interaction, amount: float, category: str = None):
    try:
        # Always load the latest data to operate on
        current_expenses_data = load_expenses()

        if amount < 0:
            await interaction.response.send_message("The amount must be non-negative.")
            return

        response_msg = "" # Initialize response message

        if category:
            # If a category is specified, only update the initial total for that category
            # and DO NOT affect current_balance.
            if category == "income":
                current_expenses_data["initial_income_total"] += amount
                response_msg = f"Đã thêm **{amount:,.0f} VNĐ** vào tổng thu nhập ban đầu."
            elif category == "outcome":
                current_expenses_data["initial_outcome_total"] += amount
                response_msg = f"Đã thêm **{amount:,.0f} VNĐ** vào tổng chi tiêu ban đầu."
            elif category == "saving":
                current_expenses_data["initial_savings_total"] += amount
                response_msg = f"Đã thêm **{amount:,.0f} VNĐ** vào tổng tiết kiệm ban đầu."
            else:
                await interaction.response.send_message("Loại danh mục không hợp lệ. Vui lòng chọn 'Income', 'Outcome', hoặc 'Saving'.")
                return
        else:
            # If no category is specified, ADD the amount to the current_balance.
            current_expenses_data["current_balance"] += amount # Change from = to +=
            response_msg = f"Đã thêm **{amount:,.0f} VNĐ** vào số dư tiền mặt của bạn. Số dư hiện tại: **{current_expenses_data['current_balance']:,.0f} VNĐ**"

        save_expenses(current_expenses_data) # Save the updated data

        # IMPORTANT: Reload the global expenses_data after saving from the function
        # This ensures other commands see the very latest state.
        global expenses_data
        expenses_data = load_expenses()

        await interaction.response.send_message(response_msg)
        print(f"Input command executed. Category: {category}, Amount: {amount}. Current balance: {expenses_data['current_balance']}") # Debug print
    except Exception as e:
        await interaction.response.send_message(f"Có lỗi xảy ra khi xử lý lệnh /input: {e}")
        print(f"Error in /input command: {e}") # Debug print for command error

# Run the bot
bot.run(TOKEN)