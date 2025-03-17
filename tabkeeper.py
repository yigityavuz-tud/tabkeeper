import boto3
import json
import re
import pandas as pd
from datetime import datetime, timezone
import ast
from io import StringIO

from openai import OpenAI

# O stands for class functions that interact with an object's attributes. They tend to return None
# C stands for functions that are contained within its own. They tend to return values

# Function to extract date and hour from a text
def c_extract_date_hour(text):
    # Regular expressions for date and time
    date_pattern = r"(\d{2}\.\d{2}\.\d{2,4})"  # Matches dd.mm.yy or dd.mm.yyyy
    time_pattern = r"(\d{2}:\d{2}(?::\d{2})?)"  # Matches hh:mm or hh:mm:ss

    p_date, p_hour = None, None

    # Search for date
    date_match = re.search(date_pattern, text)
    if date_match:
        date_str = date_match.group(1)
        try:
            # Try parsing with 4-digit year first
            p_date = datetime.strptime(date_str, "%d.%m.%Y").date()
        except ValueError:
            # Fallback to 2-digit year
            p_date = datetime.strptime(date_str, "%d.%m.%y").date()

    # Search for time
    time_match = re.search(time_pattern, text)
    if time_match:
        time_str = time_match.group(1)
        try:
            # Try parsing with seconds first
            p_hour = datetime.strptime(time_str, "%H:%M:%S").time()
        except ValueError:
            # Fallback to hours and minutes only
            p_hour = datetime.strptime(time_str, "%H:%M").time()

    return p_date, p_hour

# Function to extract article names and prices
def c_receipt_items_processor(receipt, store_brand):
    article_names = []  # List to store the resulting article names
    article_prices = []  # List to store the resulting article prices
    skip_count = 0  # Counter for consecutive invalid pairs
    start_index = -1  # Index to start processing from

    # Find the first occurrence of "eur"
    for i, item in enumerate(receipt):
        if item == "eur":
            start_index = i + 1
            break

    # If "eur" is not found, return an empty list
    if start_index == -1:
        return article_names, article_prices

    # Process elements in pairs
    i = start_index
    while i < len(receipt) - 1 and skip_count < 4:
        # Get the pair
        article_part = receipt[i]
        price_part = receipt[i + 1]

        # Process the article name
        article_parts = article_part.split(" ")
        if store_brand == "aldi süd":
            article_name = " ".join(article_parts[1:])  # Remove the first element and join the rest if the store brand is aldi süd
        else:
            article_name = article_part

        # Process the price
        price_parts = price_part.split(" ")
        if len(price_parts) >= 1:
            price_value = price_parts[0]  # The price is the first part
            price_suffix = price_parts[-1] if len(price_parts) > 1 else ""  # The suffix (e.g., "a" or "b")

            # Validate the price
            if (
                len(price_suffix) == 1  # Suffix must be a single letter
                and price_value.count(",") == 1  # Price must contain exactly one comma
                and all(c.isdigit() or c == "," for c in price_value)  # Only digits and comma allowed
            ):
                # Handle leading comma
                if price_value.startswith(","):
                    price_value = "0" + price_value
                # Convert to float
                try:
                    price = float(price_value.replace(",", "."))
                    # Add the valid results to the respective lists
                    article_names.append(article_name)
                    article_prices.append(price)
                    skip_count = 0  # Reset the skip counter
                except ValueError:
                    skip_count += 1  # Invalid price format
            else:
                skip_count += 1  # Invalid price format
        else:
            skip_count += 1  # Invalid price format

        # Move to the next item, not the next pair
        # If the price name and price order is mixed up, the pair will be eliminated by the price conditions
        # That's worth the additional computational load
        i += 1

    return article_names, article_prices

class BonEngine:
    def __init__(self, config):
        # Sanitize the config values by stripping whitespace and other unwanted characters
        self.bucket_name = config["bucket_name"].strip()  # Remove leading/trailing spaces
        self.pic_format = config["pic_format"].strip()
        self.api_key = config["api_key"].strip()
        self.base_url = config["base_url"].strip()
        self.prompt_1 = config["prompt_1"].strip()
        self.prompt_2 = config["prompt_2"].strip()
        self.prompt_3 = config["prompt_3"].strip()
        self.output_filename = config["output_filename"].strip()
        self.verbose = config["verbose"]  # Optional, default to False

        self.ai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Construct the S3 path after sanitizing
        self.s3_path = f"s3://{self.bucket_name}/{self.output_filename}"

        # Initialize AWS clients
        self.s3 = boto3.client('s3')
        self.textract = boto3.client('textract')

        # Initialize other attributes
        self.bucket = self.s3.list_objects_v2(Bucket=self.bucket_name)
        self.receipts = []
        self.receipts_raw_text = []
        self.receipt_texts = []
        self.store_names = []
        self.receipts_dict = {}
        self.df_dict = {}
        self.df = None
        self.existing_df = None
        self.row_ranges = []
        self.distinct_receipt_codes = []
        self.item_categories = []
        self.unique_itemlist = []
        self.item_category_map = {}

    
    # Function to import receipts from an S3 bucket
    def o_import_receipt_names(self):
        while True:
            bucket = self.s3.list_objects_v2(Bucket=self.bucket_name)["Contents"]
            for i in range(1, len(bucket)):
                if bucket[i]["Key"][-5:] == self.pic_format:
                    self.receipts.append(bucket[i]["Key"])
            break
        return None
    
    def o_image_to_text(self):
        for receipt in self.receipts:
            if self.verbose:  # Check if verbose mode is enabled
                print(f"Processing receipt: {receipt}")
    
            response = self.textract.analyze_document(
                Document={'S3Object': {'Bucket': self.bucket_name, 'Name': receipt}},
                FeatureTypes=['FORMS']  # You can also use 'TABLES' for structured data
            )
            self.receipts_raw_text.append(response)
        return None
    
    # Function to extract text from the Textract response
    def o_extract_text_from_response(self):
        for i in range(0, len(self.receipts_raw_text)):
            temp_list = []
            for j in range(1, len(self.receipts_raw_text[i]["Blocks"])):
                try:
                    # Print for debugging purposes
                    # print(i,j)
                    if self.receipts_raw_text[i]["Blocks"][j]["BlockType"] == "LINE":
                        temp_list.append(self.receipts_raw_text[i]["Blocks"][j]["Text"].lower())
                except:
                    continue
            self.receipt_texts.append(temp_list)
        return None
    
    
    # Function to derive store names
    def o_get_store_brands(self):
        self.receipts_dict = dict(zip(self.receipts, self.receipt_texts))
        # Derive store names
        for key, value in self.receipts_dict.items():
            temp_indicator = 0
            for i in range(0, 5):
                receipt_content = value[i].replace(" ", "")
                if "rewe" in receipt_content:
                    # print(key, value[i], "rewe")
                    self.store_names.append("rewe")
                    temp_indicator = 1
                    pass
                elif "aldi" in receipt_content or "süd" in receipt_content:
                    # print(key, value[i], "aldi süd")
                    self.store_names.append("aldi süd")
                    temp_indicator = 1
                    pass
            if temp_indicator == 0:
                # print(key, value[i], "lidl")
                self.store_names.append("lidl")
        return None
    
    # Function to execute the extraction functions that yield purchased items and purchase date and time by receipt
    def o_exctract_features(self):
        # Prepare df_dict to be processed
        self.df_dict = {"receipt_code": [], "store_brand": [], "text": [], "purchase_date": [], "purchase_hour": [], "purchased_item": [], "item_price": []}
        self.df_dict["receipt_code"] = self.receipts
        self.df_dict["store_brand"] = self.store_names
        self.df_dict["text"] = self.receipt_texts
        
        # Process the df_dict dictionary
        for i in range(len(self.df_dict["store_brand"])):
            names = []
            prices = []
            # Process receipt items
            names, prices = c_receipt_items_processor(self.df_dict["text"][i], self.df_dict["store_brand"][i])
            self. df_dict["purchased_item"].append(names)
            self.df_dict["item_price"].append(prices)
            
            # Join the list of strings in the text column into a single string
            text = " ".join(self.df_dict["text"][i])
            p_date, p_hour = c_extract_date_hour(text)
    
            # Update the purchase_date and purchase_hour columns
            self.df_dict["purchase_date"].append(p_date)
            self.df_dict["purchase_hour"].append(p_hour)
        return None
    
    # Function to create a DataFrame from the df_dict dictionary
    def o_create_df(self):
        # Remove the "text" column
        self.df_dict.pop("text")

        # Convert the dictionary to a DataFrame
        self.df = pd.DataFrame(self.df_dict)
    
        # Explode the list columns
        self.df = self.df.explode(["purchased_item", "item_price"])
    
        # Reset the index (optional, for cleaner output)
        self.df = self.df.reset_index(drop=True)
    
        # Sort the DataFrame by receipt_code
        self.df = self.df.sort_values(by="receipt_code")
    
        self.df["purchased_item_explicit"] = [None]*len(self.df)
        
        return None
        
    def o_pre_genai_prep(self):
        # Derive row ranges of receipts
        # Sort the DataFrame by receipt_code
        self.df = self.df.sort_values(by="receipt_code").reset_index(drop=True)
    
        # Get the distinct values of receipt_code
        self.distinct_receipt_codes = self.df["receipt_code"].unique()
    
        # Loop through each distinct receipt_code
        for receipt_code in self.distinct_receipt_codes:
            # Find the indices where the receipt_code occurs
            indices = self.df.index[self.df["receipt_code"] == receipt_code].tolist()
    
            # Get the start and end indices
            start_index = indices[0]  # First occurrence
            end_index = indices[-1] + 1  # Last occurrence + 1 (exclusive)
    
            # Append the result as a tuple (receipt_code, [start_index, end_index])
            self.row_ranges.append((receipt_code, [start_index, end_index]))
    
        return None
    
    # Function to get explicit item names from the AI
    def o_get_explicit_items(self):
        for i in self.row_ranges:
            print(i)
            range_start = i[1][0]
            range_end = i[1][1]
            itemlist = list(self.df.purchased_item[range_start:range_end])
            if not all(item == "" for item in itemlist):
                response = self.ai_client.chat.completions.create(
                model = "deepseek-chat",
                messages = [
                    {"role": "system", "content": self.prompt_1},
                    {"role": "user", "content": f"{itemlist}"},
                ],
                stream=False
                )
                explicit_item_names = ast.literal_eval(response.choices[0].message.content)
                self.df.loc[range_start:range_end-1, "purchased_item_explicit"] = explicit_item_names
            else:
                continue
        return None
        
    def o_stack_dfs(self):
        self.df = pd.concat([self.df, self.existing_df], ignore_index=True)
        return None

    # Function to get item categories from the AI
    def o_get_item_categories(self):
        self.unique_itemlist = list(self.df.purchased_item_explicit.unique())
        response = self.ai_client.chat.completions.create(
        model = "deepseek-chat",
        messages = [
            {"role": "system", "content": self.prompt_2},
            {"role": "user", "content": f"{self.unique_itemlist}"},
        ],
        stream=False
        )
        self.item_categories = ast.literal_eval(response.choices[0].message.content)
        return None
    
    def o_match_items_to_categories(self):
        response = self.ai_client.chat.completions.create(
        model = "deepseek-chat",
        messages = [
            {"role": "system", "content": self.prompt_3},
            {"role": "user", "content": f"Items: {self.unique_itemlist}, categories: {self.item_categories}"},
        ],
        stream=False
        )
    
        # Convert the string to a dictionary
        self.item_category_map = ast.literal_eval(response.choices[0].message.content)
    
        self.df["item_category"] = [None]*len(self.df)
    
        # Loop through the DataFrame and update the "item_category" column
        for index, row in self.df.iterrows():
            item = row["purchased_item_explicit"]
            if item in self.item_category_map:
                self.df.at[index, "item_category"] = self.item_category_map[item]
        return None
    
    def o_save_results(self):
        # Convert DataFrame to CSV in memory
        csv_buffer = StringIO()
        self.df.to_csv(csv_buffer, index=False)
    
        # Upload the CSV data to S3
        self.s3.put_object(Bucket=self.bucket_name, Key=self.output_filename, Body=csv_buffer.getvalue())
    
        print(f"DataFrame uploaded to S3 as {self.output_filename} in bucket {self.bucket_name}.")
        return None
    
    def o_sync(self):
        # Run after o_import_receipt_names
        existing_df = pd.read_csv(self.s3_path)
        existing_receipts = list(existing_df.receipt_code.unique())
        for receipt in existing_receipts:
            if receipt in self.receipts:
                self.receipts.remove(receipt)

def process_data(config):
    # Initialize the BonEngine with the provided configuration
    bon_engine = BonEngine(config)

    # Step 1: Import receipt names from S3
    bon_engine.o_import_receipt_names()

    # Check if there is existing data in the S3 bucket
    try:
        # Attempt to read the existing CSV file from S3
        bon_engine.existing_df = pd.read_csv(bon_engine.s3_path)
        print("Existing data found. Processing new receipts only.")
        
        # Sync the receipts to process only new ones
        bon_engine.o_sync()
        
        # If there are no new receipts, exit early
        if not bon_engine.receipts:
            print("No new receipts to process.")
            return

    except Exception as e:
        # If no existing data is found, process all receipts
        print("No existing data found. Processing all receipts.")
        bon_engine.existing_df = None

    # Step 2: Convert images to text using AWS Textract
    bon_engine.o_image_to_text()

    # Step 3: Extract text from the Textract response
    bon_engine.o_extract_text_from_response()

    # Step 4: Derive store brands from the receipt text
    bon_engine.o_get_store_brands()

    # Step 5: Extract features (items, prices, dates, etc.)
    bon_engine.o_exctract_features()

    # Step 6: Create a DataFrame from the extracted data
    bon_engine.o_create_df()

    # Step 7: Prepare for GenAI processing
    bon_engine.o_pre_genai_prep()

    # Step 8: Get explicit item names using AI
    bon_engine.o_get_explicit_items()

    # Step 9: If there was existing data, stack the new DataFrame with the existing one
    if bon_engine.existing_df is not None:
        bon_engine.o_stack_dfs()

    # Step 10: Get item categories using AI
    bon_engine.o_get_item_categories()

    # Step 11: Match items to their categories
    bon_engine.o_match_items_to_categories()

    # Step 12: Save the results to S3
    bon_engine.o_save_results()

    print("Data processing complete.")