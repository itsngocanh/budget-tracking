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
    """Loads expense data from a JSON file.
    If the file is not found or empty, it initializes with a default structure.
    Includes robust error handling for JSON decoding issues.
    """
    print("Attempting to load expenses data...") # Debug print
    default_data = {
        "current_balance": 0.0,
        "income": [],
        "outcome": [],
        "savings": [],
        "initial_income_total": 0.0,
        "initial_outcome_total": 0.0,
        "initial_savings_total": 0.0,
        "last_summary_date": None # Added this default key for daily summary
    }

    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        print(f"'{DATA_FILE}' not found or is empty. Initializing with default structure.") # Debug print
        # Return default data directly, no need to save here, save_expenses handles it on first write
        return default_data 
    
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Ensure all top-level keys exist and initialize new ones if missing
            for key, value in default_data.items():
                data.setdefault(key, value)
            print(f"Expenses data loaded successfully. Current balance: {data['current_balance']}") # Debug print
            return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{DATA_FILE}': {e}. Returning default structure.") # Debug print for JSON error
        # If file is corrupted, return fresh data to prevent bot crash
        return default_data
    except Exception as e:
        print(f"An unexpected error occurred while loading '{DATA_FILE}': {e}. Returning default structure.") # Debug print for other errors
        return default_data

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

# Global variable to hold expenses data, initialized on script start
expenses_data = load_expenses()

# --- Git Automation Function ---
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
            # Added set_upstream=True to explicitly set the upstream branch on the first push
            origin.push(refspec=f'{GIT_BRANCH}:{GIT_BRANCH}', set_upstream=True) 
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
    # Defer the response immediately to avoid "Unknown interaction" if Git operations take time
    await interaction.response.defer(ephemeral=False) # Use ephemeral=True if you want the response only for the user

    try:
        if amount <= 0:
            await interaction.edit_original_response(content="Số tiền thu nhập phải lớn hơn 0.")
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
        
        # Save expenses, which includes Git operations
        save_expenses(expenses_data)

        # Edit the original deferred response with the final message
        await interaction.edit_original_response(
            content=f"Đã ghi nhận khoản thu: **{amount:,.0f} VNĐ** (Loại: {subtype}, Mô tả: {description if description else 'Không có mô tả'})\n"
                    f"Số dư hiện tại: **{expenses_data['current_balance']:,.0f} VNĐ**"
        )
    except Exception as e:
        # If an error occurs, edit the original response with the error message
        await interaction.edit_original_response(content=f"Có lỗi xảy ra khi ghi nhận thu nhập: {e}")
        print(f"Error in /thu command: {e}") # Debug print for command error

@bot.tree.command(name="chi", description="Ghi nhận một khoản chi tiêu mới.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    amount="Số tiền chi tiêu (ví dụ: 50000)",
    subtype="Loại chi tiêu (ví dụ: Ăn uống, Di chuyển)",
    description="Mô tả chi tiết khoản chi tiêu (tùy chọn)"
)
async def outcome(interaction: discord.Interaction, amount: float, subtype: str, description: str = None):
    # Defer the response immediately
    await interaction.response.defer(ephemeral=False)

    try:
        if amount <= 0:
            await interaction.edit_original_response(content="Số tiền chi tiêu phải lớn hơn 0.")
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
        
        # Save expenses, which includes Git operations
        save_expenses(expenses_data)

        response_message = (
            f"Đã ghi nhận khoản chi: **{amount:,.0f} VNĐ** (Loại: {subtype}, Mô tả: {description if description else 'Không có mô tả'})\n"
            f"Số dư hiện tại: **{expenses_data['current_balance']:,.0f} VNĐ**"
        )
        if expenses_data["current_balance"] < 0:
            response_message += "\n⚠️ **Cảnh báo: Chi tiêu này đã khiến số dư của bạn âm! Vui lòng chú ý quản lý tài chính.**"
        
        await interaction.edit_original_response(content=response_message)
    except Exception as e:
        await interaction.edit_original_response(content=f"Có lỗi xảy ra khi ghi nhận chi tiêu: {e}")
        print(f"Error in /chi command: {e}") # Debug print for command error

@bot.tree.command(name="tietkiem", description="Ghi nhận một khoản tiền tiết kiệm mới và trừ vào số dư.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    amount="Số tiền tiết kiệm (ví dụ: 100000)",
    subtype="Loại tiết kiệm (ví dụ: Tiết kiệm dài hạn, Quỹ khẩn cấp)",
    description="Mô tả chi tiết khoản tiết kiệm (tùy chọn)"
)
async def saving(interaction: discord.Interaction, amount: float, subtype: str, description: str = None):
    # Defer the response immediately
    await interaction.response.defer(ephemeral=False)

    try:
        if amount <= 0:
            await interaction.edit_original_response(content="Số tiền tiết kiệm phải lớn hơn 0.")
            return
        
        # Check if enough balance for saving
        if expenses_data["current_balance"] < amount:
            await interaction.edit_original_response(
                content=f"Số dư của bạn ({expenses_data['current_balance']:,.0f} VNĐ) không đủ để tiết kiệm **{amount:,.0f} VNĐ**."
            )
            return

        print(f"Before saving: current_balance = {expenses_data['current_balance']}") # Debug print
        expenses_data["savings"].append({
            "amount": amount,
            "subtype": subtype,
            "description": description if description else "Không có mô tả",
            "timestamp": datetime.now().isoformat()
        })
        expenses_data["current_balance"] -= amount # Savings reduces cash balance
        print(f"After saving: current_balance = {expenses_data['current_balance']}") # Debug print
        
        # Save expenses, which includes Git operations
        save_expenses(expenses_data)

        response_message = (
            f"Đã chuyển **{amount:,.0f} VNĐ** vào tiết kiệm (Loại: {subtype}, Mô tả: {description if description else 'Không có mô tả'})\n"
            f"Số dư hiện tại: **{expenses_data['current_balance']:,.0f} VNĐ**"
        )
        if expenses_data["current_balance"] < 0:
            response_message += "\n⚠️ **Cảnh báo: Việc tiết kiệm này đã khiến số dư của bạn âm! Vui lòng cân nhắc kế hoạch tài chính.**"
        
        await interaction.edit_original_response(content=response_message)
    except Exception as e:
        await interaction.edit_original_response(content=f"Có lỗi xảy ra khi ghi nhận tiết kiệm: {e}")
        print(f"Error in /tietkiem command: {e}") # Debug print for command error


@bot.tree.command(name="balance", description="Hiển thị số dư tiền mặt hiện tại của bạn.", guild=discord.Object(id=GUILD_ID))
async def balance(interaction: discord.Interaction):
    """Displays the current cash balance by reloading the latest data."""
    # Defer the response in case load_expenses is slow due to file access
    await interaction.response.defer(ephemeral=False) 

    global expenses_data 
    expenses_data = load_expenses() # Reload to get the absolute latest state of the balance

    current_balance = expenses_data.get("current_balance", 0.0)
    await interaction.edit_original_response(content=f"Số dư tiền mặt hiện tại của bạn là: **{current_balance:,.0f} VNĐ**")
    print(f"Balance command executed by {interaction.user.name}. Current balance: {current_balance}.")

@bot.tree.command(name="add", description="Thêm một khoản tiền vào số dư của bạn.", guild=discord.Object(id=GUILD_ID))
async def add(interaction: discord.Interaction, amount: float):
    """Adds a specified amount to the current balance and saves/pushes data."""
    # Defer the response immediately
    await interaction.response.defer(ephemeral=False)

    if amount <= 0:
        await interaction.edit_original_response(content="Số tiền phải lớn hơn 0.")
        return

    global expenses_data
    expenses_data = load_expenses() # Ensure we're working with the freshest data

    expenses_data["current_balance"] += amount
    
    # Save expenses, which includes Git operations
    save_expenses(expenses_data)

    await interaction.edit_original_response(
        content=f"Đã thêm **{amount:,.0f} VNĐ** vào số dư của bạn. Số dư hiện tại: **{expenses_data['current_balance']:,.0f} VNĐ**"
    )
    print(f"Add command executed. Amount: {amount}. New balance: {expenses_data['current_balance']}.")

@bot.tree.command(name="subtract", description="Trừ một khoản tiền khỏi số dư của bạn.", guild=discord.Object(id=GUILD_ID))
async def subtract(interaction: discord.Interaction, amount: float):
    """Subtracts a specified amount from the current balance and saves/pushes data."""
    # Defer the response immediately
    await interaction.response.defer(ephemeral=False)

    if amount <= 0:
        await interaction.edit_original_response(content="Số tiền phải lớn hơn 0.")
        return

    global expenses_data
    expenses_data = load_expenses() # Ensure we're working with the freshest data

    if expenses_data["current_balance"] < amount:
        await interaction.edit_original_response(
            content=f"Số dư của bạn ({expenses_data['current_balance']:,.0f} VNĐ) không đủ để trừ **{amount:,.0f} VNĐ**."
        )
        return

    expenses_data["current_balance"] -= amount
    
    # Save expenses, which includes Git operations
    save_expenses(expenses_data)

    await interaction.edit_original_response(
        content=f"Đã trừ **{amount:,.0f} VNĐ** khỏi số dư của bạn. Số dư hiện tại: **{expenses_data['current_balance']:,.0f} VNĐ**"
    )
    print(f"Subtract command executed. Amount: {amount}. New balance: {expenses_data['current_balance']}.")

@bot.tree.command(name="input", description="Ghi nhận thu nhập, chi tiêu, hoặc tiết kiệm.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.choices(category=[
    discord.app_commands.Choice(name="Income (Thu nhập)", value="income"),
    discord.app_commands.Choice(name="Outcome (Chi tiêu)", value="outcome"),
    discord.app_commands.Choice(name="Saving (Tiết kiệm)", value="saving"),
    discord.app_commands.Choice(name="Initial Income (Tổng thu nhập ban đầu)", value="initial_income"),
    discord.app_commands.Choice(name="Initial Outcome (Tổng chi tiêu ban đầu)", value="initial_outcome"),
    discord.app_commands.Choice(name="Initial Saving (Tổng tiết kiệm ban đầu)", value="initial_saving")
])
async def input_expense(
    interaction: discord.Interaction,
    amount: float, # Changed to float for consistency, previously was int in some places
    category: discord.app_commands.Choice[str],
    subtype: str = None,
    description: str = None
):
    """
    Records income, outcome, or savings transactions, or updates initial budget totals.
    Args:
        amount (float): The amount of money involved in the transaction.
        category (str): The type of transaction (income, outcome, saving, initial_income, initial_outcome, initial_saving).
        subtype (str, optional): A specific sub-category for the transaction (e.g., "Food", "Salary").
        description (str, optional): A brief note or description about the transaction.
    """
    await interaction.response.defer(ephemeral=False) # Ensure ephemeral is consistent, previously was True

    if amount <= 0:
        await interaction.edit_original_response(content="Số tiền phải lớn hơn 0.")
        return

    global expenses_data
    current_expenses_data = load_expenses() # Always load the freshest data to prevent race conditions

    timestamp = datetime.now().isoformat() # ISO format for easy sorting and parsing

    response_msg = ""

    # Handle initial balance updates separately (these do not affect current_balance directly)
    if category.value.startswith("initial_"):
        if category.value == "initial_income":
            current_expenses_data["initial_income_total"] += amount
            response_msg = f"Đã thêm **{amount:,.0f} VNĐ** vào tổng thu nhập ban đầu."
        elif category.value == "initial_outcome":
            current_expenses_data["initial_outcome_total"] += amount
            response_msg = f"Đã thêm **{amount:,.0f} VNĐ** vào tổng chi tiêu ban đầu."
        elif category.value == "initial_saving":
            current_expenses_data["initial_savings_total"] += amount
            response_msg = f"Đã thêm **{amount:,.0f} VNĐ** vào tổng tiết kiệm ban đầu."
        else:
            await interaction.edit_original_response(content="Loại danh mục ban đầu không hợp lệ.")
            return
    else:
        # Handle regular income, outcome, saving transactions
        entry = {
            "amount": amount,
            "subtype": subtype if subtype else "Không có", # Default subtype
            "description": description if description else "Không có", # Default description
            "timestamp": timestamp,
            "user": interaction.user.name # Record which user made the entry
        }

        if category.value == "income":
            current_expenses_data["income"].append(entry)
            current_expenses_data["current_balance"] += amount # Income adds to current balance
            response_msg = f"Đã ghi nhận **{amount:,.0f} VNĐ** vào thu nhập. Số dư hiện tại: **{current_expenses_data['current_balance']:,.0f} VNĐ**"
        elif category.value == "outcome":
            if current_expenses_data["current_balance"] < amount:
                await interaction.edit_original_response(content=f"Số dư của bạn ({current_expenses_data['current_balance']:,.0f} VNĐ) không đủ để chi tiêu **{amount:,.0f} VNĐ**.")
                return
            current_expenses_data["outcome"].append(entry)
            current_expenses_data["current_balance"] -= amount # Outcome subtracts from current balance
            response_msg = f"Đã ghi nhận **{amount:,.0f} VNĐ** vào chi tiêu. Số dư hiện tại: **{current_expenses_data['current_balance']:,.0f} VNĐ**"
        elif category.value == "saving":
            if current_expenses_data["current_balance"] < amount:
                await interaction.edit_original_response(content=f"Số dư của bạn ({current_expenses_data['current_balance']:,.0f} VNĐ) không đủ để tiết kiệm **{amount:,.0f} VNĐ**.")
                return
            current_expenses_data["savings"].append(entry)
            current_expenses_data["current_balance"] -= amount # Savings reduce cash balance (moving money from cash to savings)
            response_msg = f"Đã ghi nhận **{amount:,.0f} VNĐ** vào tiết kiệm. Số dư hiện tại: **{current_expenses_data['current_balance']:,.0f} VNĐ**"
        else:
            await interaction.edit_original_response(content="Loại danh mục không hợp lệ. Vui lòng chọn 'Income', 'Outcome', hoặc 'Saving'.")
            return

    save_expenses(current_expenses_data) # Save the updated data, which automatically triggers Git push

    # IMPORTANT: Reload the global expenses_data after saving from the function
    # This ensures other commands see the very latest state of the data in memory.
    expenses_data = load_expenses()

    await interaction.edit_original_response(content=response_msg)
    print(f"Input command executed. Category: {category.value}, Amount: {amount}. Current balance: {expenses_data['current_balance']}.")


@bot.tree.command(name="export_csv", description="Xuất dữ liệu thu chi và tiết kiệm ra file CSV.", guild=discord.Object(id=GUILD_ID))
async def export_csv(interaction: discord.Interaction):
    """Exports all transaction data to a CSV file and sends it as a Discord attachment.
    This is for manual, on-demand export to Discord, distinct from the daily automated export.
    """
    # Defer the response in case CSV conversion takes time
    await interaction.response.defer(ephemeral=False)

    csv_filename = "03.my_data.csv"
    global expenses_data
    expenses_data = load_expenses() # Ensure freshest data for export

    if convert_json_to_csv(expenses_data, csv_filename):
        try:
            await interaction.edit_original_response(
                content="Đây là dữ liệu thu chi và tiết kiệm của bạn:",
                file=discord.File(csv_filename)
            )
            os.remove(csv_filename) # Clean up the local file after sending to Discord
            # Note: No need to call push_to_github here as save_expenses already handles pushing 03.my_data.csv
            # when data is modified. This command is purely for sending the file to Discord.
        except Exception as e:
            await interaction.edit_original_response(content=f"Có lỗi xảy ra khi xuất CSV: {e}")
            print(f"Error sending CSV file via /export_csv command: {e}")
    else:
        await interaction.edit_original_response(content="Có lỗi xảy ra khi tạo file CSV.")
        print("Error creating CSV file for /export_csv command.")

@bot.tree.command(name="total", description="Hiển thị tổng thu nhập, chi tiêu và tiết kiệm ban đầu.", guild=discord.Object(id=GUILD_ID))
async def total(interaction: discord.Interaction):
    """Displays initial total income, outcome, and savings as configured by 'initial_' input commands."""
    # Defer the response for consistency
    await interaction.response.defer(ephemeral=False)

    global expenses_data
    expenses_data = load_expenses() # Load the latest data

    initial_income = expenses_data.get("initial_income_total", 0.0)
    initial_outcome = expenses_data.get("initial_outcome_total", 0.0)
    initial_savings = expenses_data.get("initial_savings_total", 0.0)

    # Create an embedded message for a clean display of initial totals
    embed = discord.Embed(
        title="Tổng Quan Ban Đầu",
        description="Tổng thu nhập, chi tiêu và tiết kiệm được ghi nhận ban đầu (không thay đổi bởi giao dịch hàng ngày).",
        color=discord.Color.green()
    )
    embed.add_field(name="Tổng Thu Nhập Ban Đầu", value=f"**{initial_income:,.0f} VNĐ**", inline=False)
    embed.add_field(name="Tổng Chi Tiêu Ban Đầu", value=f"**{initial_outcome:,.0f} VNĐ**", inline=False)
    embed.add_field(name="Tổng Tiết Kiệm Ban Đầu", value=f"**{initial_savings:,.0f} VNĐ**", inline=False)
    embed.set_footer(text="Các số liệu này chỉ được cập nhật qua lệnh /input với danh mục 'Initial...'")

    await interaction.edit_original_response(embed=embed)
    print(f"Total command executed by {interaction.user.name}.")

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
            income_lines.append("Không có khoản thu nào.\n") # Added \n for consistent spacing
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
            outcome_lines.append("Không có khoản chi nào.\n") # Added \n for consistent spacing
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
            saving_lines.append("Không có khoản tiết kiệm nào.\n") # Added \n for consistent spacing
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
        # +1 for the newline character that will be added when joining lines later
        if current_chunk_length + len(line) + 1 > MAX_CHUNK_LENGTH and current_chunk: 
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

@bot.tree.command(name="clear_expense", description="Chọn và xóa một khoản thu, chi, hoặc tiết kiệm cụ thể.", guild=discord.Object(id=GUILD_ID))
@discord.app_commands.describe(
    expense_type="Loại khoản cần xóa (income/outcome/saving)",
    index="Chỉ số chính xác của khoản cần xóa (dùng lệnh '/list_expenses' để xem)"
)
async def clear_expense(interaction: discord.Interaction, expense_type: str, index: int):
    # Defer the response immediately as this involves data modification and Git ops
    await interaction.response.defer(ephemeral=False)

    expense_type_lower = expense_type.lower()
    if expense_type_lower not in ["income", "outcome", "saving"]:
        await interaction.edit_original_response(content="Loại khoản không hợp lệ. Vui lòng chọn 'income', 'outcome', hoặc 'saving'.")
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
        await interaction.edit_original_response(content="Lỗi nội bộ: Không thể ánh xạ loại khoản chi phí.")
        return

    target_list = current_expenses_data[actual_key]

    # Check if the provided index is within the valid range of the actual list
    if not (0 <= index < len(target_list)): # Index is now 0-based
        await interaction.edit_original_response(content="Chỉ số không hợp lệ hoặc khoản không tồn tại. Vui lòng kiểm tra lại bằng lệnh '/list_expenses' mới nhất.")
        return

    try:
        removed_item = target_list.pop(index) # Use the index directly

        # Adjust current_balance based on the type of removed item
        if expense_type_lower == "income":
            current_expenses_data["current_balance"] -= removed_item["amount"]
        elif expense_type_lower == "outcome": # Outcome removal increases balance
            current_expenses_data["current_balance"] += removed_item["amount"]
        elif expense_type_lower == "saving": # Saving removal increases balance (money returns to cash)
            current_expenses_data["current_balance"] += removed_item["amount"]

        # Save the updated data, which triggers Git push
        save_expenses(current_expenses_data)

        # Reload the global expenses_data after saving from the function
        global expenses_data
        expenses_data = load_expenses()

        await interaction.edit_original_response(
            content=f"Đã xóa khoản **{expense_type_lower}**:\n"
            f"**Số tiền:** {removed_item['amount']:,.0f} VNĐ\n"
            f"**Loại:** {removed_item['subtype']}\n"
            f"**Mô tả:** {removed_item['description']}\n"
            f"**Thời gian:** {datetime.fromisoformat(removed_item['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Số dư hiện tại: **{expenses_data['current_balance']:,.0f} VNĐ**"
        )
        print(f"Clear expense command executed. Type: {expense_type_lower}, Index: {index}.") # Debug print
    except Exception as e:
        await interaction.edit_original_response(content=f"Có lỗi xảy ra khi xóa khoản: {e}")
        print(f"Error in /clear_expense command: {e}") # Debug print for command error


# Entry point for running the bot
if __name__ == '__main__':
    # Initial load of data when the script starts
    expenses_data = load_expenses()
    bot.run(TOKEN)