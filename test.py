# import os
# import logging
# import csv
# import pdfplumber
# import shutil
# import time
# from langchain_openai import AzureChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Set up Azure OpenAI credentials
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
# AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-mini"

# # Initialize the Azure OpenAI model
# llm = AzureChatOpenAI(
#     azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     openai_api_key=AZURE_OPENAI_API_KEY,
#     openai_api_version=AZURE_OPENAI_API_VERSION,
#     temperature=0.1
# )

# # Directories setup
# INPUT_DIR = "input"
# OUTPUT_DIR = "output"
# ARCHIVE_DIR = "archive"

# os.makedirs(INPUT_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(ARCHIVE_DIR, exist_ok=True)

# # Configure logging to save only time-based logs
# log_file_path = os.path.join(OUTPUT_DIR, "process_logs.txt")

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.StreamHandler(),  # Display logs in the console
#         logging.FileHandler(log_file_path, mode='w', encoding='utf-8')  # Save logs to a text file
#     ]
# )

# # Counter for auto-incrementing ID
# id_counter = -1

# def extract_pdf_text_and_tables(pdf_path):
#     """Extract text and tables from a PDF."""
#     full_text = ""
#     all_tables = []

#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             raw_text = page.extract_text(x_tolerance=2, y_tolerance=2)
#             full_text += raw_text if raw_text else ""

#             tables = page.extract_tables(
#                 table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"}
#             )
#             if tables:
#                 all_tables.extend(tables)

#     return full_text, all_tables

# def analyze_with_llm(raw_text, tables_text):
#     prompt = (
#         "You are an expert in extracting structured data from tool specification documents. "
#         "Analyze the following text and extract key specifications for each tool model mentioned. "
#         "For each tool model, extract the following details:\n"
#         "- **Product_Usage**: Information about product usage\n"
#         "- **Product_Sub_Category**: Information about Sub-category of product\n"
#         "- **Product_Name**: Information about product name\n"
#         "- **Product_Description**: Description of product\n"
#         "- **Product_Configuration_or_Specifications**: Configuration/Specification of product\n"
#         "- **Product_Variants**: Different Variants of the product\n"
#         "- **Product_Colors**: Colors of the product\n"
#         "- **Other_Features_or_attributes**: Features/Attributes of product\n"
#         "- **Image_URLs**: Image links from the pro\n"
#         "If a model is mentioned in a table or paragraph, ensure all following data is grouped under that model.\n\n"
#         "For any of the above categories, if the information is presented in a table, ensure that the data is correctly associated with the corresponding model. No assumption, no approximate required.\n\n"
#         f"Text extracted:\n{raw_text}\n\nTables:\n{tables_text}"
#     )

#     try:
#         messages = [
#             SystemMessage(content="You are a technical expert extracting structured data from tool specifications."),
#             HumanMessage(content=prompt)
#         ]

#         response = llm.invoke(messages)

#         llm_output = response.content.strip()
#         logging.debug(f"LLM Response:\n{llm_output}")  # Debugging, but not in the log file

#         return llm_output
#     except Exception as e:
#         logging.error(f"Error calling Azure OpenAI API: {e}")
#         return None

# headers = ["Product_Usage", "id", "Product_Sub_Category", "Product_Name", "Product_Description", "Product_Configuration_or_Specifications", "Product_Variants", "Product_Colors", "Other_Features_or_attributes", "Image_URLs"]

# def parse_llm_output(llm_output, raw_text, headers):
#     global id_counter  # Use the global counter for auto-incrementing ID
#     extracted_data = []
#     models = llm_output.split("\n\n")  # Split by double newlines to separate each model's data
#     total_columns = len(headers)  # Total columns dynamically based on headers

#     for model in models:
#         model_data = {}
#         lines = model.split("\n")

#         for line in lines:
#             if ": " in line:
#                 key, value = line.split(": ", 1)
#                 key = key.strip().replace("**", "").strip()  # Remove markdown formatting
#                 value = value.strip()

#                 # Map the key to the correct column name
#                 if "Product_Usage" in key:
#                     model_data["Product_Usage"] = value
#                 elif "Product_Sub_Category" in key:
#                     model_data["Product_Sub_Category"] = value
#                 elif "Product_Name" in key:
#                     model_data["Product_Name"] = value
#                 elif "Product_Description" in key:
#                     model_data["Product_Description"] = value
#                 elif "Product_Configuration_or_Specifications" in key:
#                     model_data["Product_Configuration_or_Specifications"] = value
#                 elif "Product_Variants" in key:
#                     model_data["Product_Variants"] = value
#                 elif "Product_Colors" in key:
#                     model_data["Product_Colors"] = value
#                 elif "Other_Features_or_attributes" in key:
#                     model_data["Other_Features_or_attributes"] = value
#                 elif "Image_URLs" in key:
#                     model_data["Image_URLs"] = value

#         # Add auto-incrementing ID
#         model_data["id"] = str(id_counter)
#         id_counter += 1

#         # Check if at least 50% of the columns have data
#         filled_columns = sum(1 for key in model_data if model_data[key])
#         if filled_columns >= total_columns * 0.5:  # At least 50% of columns have data
#             extracted_data.append(model_data)

#     return extracted_data

# def save_as_csv(parsed_data, output_file_path):
#     with open(output_file_path, "w", newline="", encoding="utf-8") as csvfile:
#         writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)  # Ensure quoting is enabled for multiline fields
#         writer.writerow(headers)

#         for row in parsed_data:
#             # Write row with proper formatting for multiline fields
#             writer.writerow([
#                 row.get("Product_Usage", ""),
#                 row.get("id", ""),
#                 row.get("Product_Sub_Category", ""),
#                 row.get("Product_Name", ""),
#                 row.get("Product_Description", ""),
#                 row.get("Product_Configuration_or_Specifications", ""),
#                 row.get("Product_Variants", ""),
#                 row.get("Product_Colors", ""),
#                 row.get("Other_Features_or_attributes", ""),
#                 row.get("Image_URLs", ""),
#             ])

#     logging.info(f"CSV file saved: {output_file_path}")

# def process_pdf(pdf_path):
#     """Process a single PDF file."""
#     start_time = time.time()  # Start timer for processing

#     try:
#         logging.info(f"Started processing PDF: {pdf_path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
#         # Start timer for reading file
#         read_start_time = time.time()
#         raw_text, tables = extract_pdf_text_and_tables(pdf_path)
#         read_end_time = time.time()
#         read_duration = read_end_time - read_start_time
#         logging.info(f"Reading PDF took {read_duration:.2f} seconds.")
        
#         tables_text = "\n".join(
#             [" | ".join(map(str, row)) for table in tables for row in table]
#         ) if tables else ""

#         # Start timer for LLM invocation
#         llm_start_time = time.time()
#         analyzed_text = analyze_with_llm(raw_text, tables_text)
#         llm_end_time = time.time()
#         llm_duration = llm_end_time - llm_start_time
#         logging.info(f"LLM invocation took {llm_duration:.2f} seconds.")
        
#         if not analyzed_text:
#             raise ValueError("No response from Azure OpenAI")

#         structured_data = parse_llm_output(analyzed_text, raw_text, headers)
#         if not structured_data:
#             logging.warning(f"Skipping {pdf_path} as less than 50% of the columns have data.")
#             return

#         # Start timer for writing CSV file
#         csv_start_time = time.time()
#         output_filename = os.path.join(OUTPUT_DIR, os.path.basename(pdf_path).replace(".pdf", "_output.csv"))
#         save_as_csv(structured_data, output_filename)
#         csv_end_time = time.time()
#         csv_duration = csv_end_time - csv_start_time
#         logging.info(f"Writing CSV file took {csv_duration:.2f} seconds.")
        
#         shutil.move(pdf_path, os.path.join(ARCHIVE_DIR, os.path.basename(pdf_path)))
#         logging.info(f"PDF processed and archived: {pdf_path}")

#         end_time = time.time()  # End timer for processing
#         total_duration = end_time - start_time
#         logging.info(f"Total processing time for {pdf_path} was {total_duration:.2f} seconds.")

#     except Exception as e:
#         logging.error(f"Error processing {pdf_path}: {e}")

# def main():
#     """Main function to process all PDFs in the input directory."""
#     start_time = time.time()  # Start timer for the whole process
#     logging.info("Starting PDF processing...")

#     for pdf_file in os.listdir(INPUT_DIR):
#         if pdf_file.endswith(".pdf"):
#             process_pdf(os.path.join(INPUT_DIR, pdf_file))
    
#     end_time = time.time()  # End timer for the whole process
#     total_duration = end_time - start_time
#     logging.info(f"Total time for all PDF processing: {total_duration:.2f} seconds.")

# if __name__ == "__main__":
#     main()





# ----------------- Code with saving prcessing logs -------------
# import os
# import logging
# import csv
# import pdfplumber
# import shutil
# import time
# from langchain_openai import AzureChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Set up Azure OpenAI credentials
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
# AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-mini"

# # Initialize the Azure OpenAI model
# llm = AzureChatOpenAI(
#     azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     openai_api_key=AZURE_OPENAI_API_KEY,
#     openai_api_version=AZURE_OPENAI_API_VERSION,
#     temperature=0.1
# )

# # Directories setup
# INPUT_DIR = "input"
# OUTPUT_DIR = "output"
# ARCHIVE_DIR = "archive"
# DUMP_DIR = "dump"  # Directory for files with less than 50% data

# os.makedirs(INPUT_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(ARCHIVE_DIR, exist_ok=True)
# os.makedirs(DUMP_DIR, exist_ok=True)  # Create dump folder

# # Configure logging to save only time-based logs
# log_file_path = os.path.join(OUTPUT_DIR, "process_logs.txt")

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.StreamHandler(),  # Display logs in the console
#         logging.FileHandler(log_file_path, mode='w', encoding='utf-8')  # Save logs to a text file
#     ]
# )

# # Counter for auto-incrementing ID
# id_counter = 0

# def extract_pdf_text_and_tables(pdf_path):
#     """Extract text and tables from a PDF."""
#     full_text = ""
#     all_tables = []

#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             raw_text = page.extract_text(x_tolerance=2, y_tolerance=2)
#             full_text += raw_text if raw_text else ""

#             tables = page.extract_tables(
#                 table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"}
#             )
#             if tables:
#                 all_tables.extend(tables)

#     return full_text, all_tables

# def analyze_with_llm(raw_text, tables_text):
#     prompt = (
#         "You are an expert in extracting structured data from tool specification documents. "
#         "Analyze the following text and extract key specifications for each tool model mentioned. "
#         "For each tool model, extract the following details:\n"
#         "- **Product_Usage**: Information about product usage\n"
#         "- **Product_Sub_Category**: Information about Sub-category of product\n"
#         "- **Product_Name**: Information about product name\n"
#         "- **Product_Description**: Description of product\n"
#         "- **Product_Configuration_or_Specifications**: Configuration/Specification of product\n"
#         "- **Product_Variants**: Different Variants of the product\n"
#         "- **Product_Colors**: Colors of the product\n"
#         "- **Other_Features_or_attributes**: Features/Attributes of product\n"
#         "- **Image_URLs**: Image links from the pro\n"
#         "If a model is mentioned in a table or paragraph, ensure all following data is grouped under that model.\n\n"
#         "For any of the above categories, if the information is presented in a table, ensure that the data is correctly associated with the corresponding model. No assumption, no approximate required.\n\n"
#         f"Text extracted:\n{raw_text}\n\nTables:\n{tables_text}"
#     )

#     try:
#         messages = [
#             SystemMessage(content="You are a technical expert extracting structured data from tool specifications."),
#             HumanMessage(content=prompt)
#         ]

#         response = llm.invoke(messages)

#         llm_output = response.content.strip()
#         logging.debug(f"LLM Response:\n{llm_output}")  # Debugging, but not in the log file

#         return llm_output
#     except Exception as e:
#         logging.error(f"Error calling Azure OpenAI API: {e}")
#         return None

# headers = ["Product_Usage", "id", "Product_Sub_Category", "Product_Name", "Product_Description", "Product_Configuration_or_Specifications", "Product_Variants", "Product_Colors", "Other_Features_or_attributes", "Image_URLs"]

# def parse_llm_output(llm_output, raw_text, headers):
#     global id_counter  # Use the global counter for auto-incrementing ID
#     extracted_data = []
#     models = llm_output.split("\n\n")  # Split by double newlines to separate each model's data
#     total_columns = len(headers)  # Total columns dynamically based on headers

#     for model in models:
#         model_data = {}
#         lines = model.split("\n")

#         for line in lines:
#             if ": " in line:
#                 key, value = line.split(": ", 1)
#                 key = key.strip().replace("**", "").strip()  # Remove markdown formatting
#                 value = value.strip()

#                 # Map the key to the correct column name
#                 if "Product_Usage" in key:
#                     model_data["Product_Usage"] = value
#                 elif "Product_Sub_Category" in key:
#                     model_data["Product_Sub_Category"] = value
#                 elif "Product_Name" in key:
#                     model_data["Product_Name"] = value
#                 elif "Product_Description" in key:
#                     model_data["Product_Description"] = value
#                 elif "Product_Configuration_or_Specifications" in key:
#                     model_data["Product_Configuration_or_Specifications"] = value
#                 elif "Product_Variants" in key:
#                     model_data["Product_Variants"] = value
#                 elif "Product_Colors" in key:
#                     model_data["Product_Colors"] = value
#                 elif "Other_Features_or_attributes" in key:
#                     model_data["Other_Features_or_attributes"] = value
#                 elif "Image_URLs" in key:
#                     model_data["Image_URLs"] = value

#         # Add auto-incrementing ID
#         model_data["id"] = str(id_counter)
#         id_counter += 1

#         # Check if at least 50% of the columns have data
#         filled_columns = sum(1 for key in model_data if model_data[key])
#         if filled_columns >= total_columns * 0.5:  # At least 50% of columns have data
#             extracted_data.append(model_data)

#     return extracted_data

# def save_as_csv(parsed_data, output_file_path):
#     with open(output_file_path, "w", newline="", encoding="utf-8") as csvfile:
#         writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)  # Ensure quoting is enabled for multiline fields
#         writer.writerow(headers)

#         for row in parsed_data:
#             # Write row with proper formatting for multiline fields
#             writer.writerow([ 
#                 row.get("Product_Usage", ""),
#                 row.get("id", ""),
#                 row.get("Product_Sub_Category", ""),
#                 row.get("Product_Name", ""),
#                 row.get("Product_Description", ""),
#                 row.get("Product_Configuration_or_Specifications", ""),
#                 row.get("Product_Variants", ""),
#                 row.get("Product_Colors", ""),
#                 row.get("Other_Features_or_attributes", ""),
#                 row.get("Image_URLs", ""),
#             ])

#     logging.info(f"CSV file saved: {output_file_path}")

# def process_pdf(pdf_path):
#     """Process a single PDF file."""
#     start_time = time.time()  # Start timer for processing

#     try:
#         logging.info(f"Started processing PDF: {pdf_path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
#         # Start timer for reading file
#         read_start_time = time.time()
#         raw_text, tables = extract_pdf_text_and_tables(pdf_path)
#         read_end_time = time.time()
#         read_duration = read_end_time - read_start_time
#         logging.info(f"Reading PDF took {read_duration:.2f} seconds.")
        
#         tables_text = "\n".join(
#             [" | ".join(map(str, row)) for table in tables for row in table]
#         ) if tables else ""

#         # Start timer for LLM invocation
#         llm_start_time = time.time()
#         analyzed_text = analyze_with_llm(raw_text, tables_text)
#         llm_end_time = time.time()
#         llm_duration = llm_end_time - llm_start_time
#         logging.info(f"LLM invocation took {llm_duration:.2f} seconds.")
        
#         if not analyzed_text:
#             raise ValueError("No response from Azure OpenAI")

#         structured_data = parse_llm_output(analyzed_text, raw_text, headers)
#         if not structured_data:
#             logging.warning(f"Skipping {pdf_path} as less than 50% of the columns have data.")
#             shutil.move(pdf_path, os.path.join(DUMP_DIR, os.path.basename(pdf_path)))  # Move to dump folder
#             return

#         # Start timer for writing CSV file
#         csv_start_time = time.time()
#         output_filename = os.path.join(OUTPUT_DIR, os.path.basename(pdf_path).replace(".pdf", "_output.csv"))
#         save_as_csv(structured_data, output_filename)
#         csv_end_time = time.time()
#         csv_duration = csv_end_time - csv_start_time
#         logging.info(f"Writing CSV file took {csv_duration:.2f} seconds.")
        
#         shutil.move(pdf_path, os.path.join(ARCHIVE_DIR, os.path.basename(pdf_path)))
#         logging.info(f"PDF processed and archived: {pdf_path}")

#         end_time = time.time()  # End timer for processing
#         total_duration = end_time - start_time
#         logging.info(f"Total processing time for {pdf_path} was {total_duration:.2f} seconds.")

#     except Exception as e:
#         logging.error(f"Error processing {pdf_path}: {e}")

# def main():
#     """Main function to process all PDFs in the input directory."""
#     start_time = time.time()  # Start timer for the whole process
#     logging.info("Starting PDF processing...")

#     for pdf_file in os.listdir(INPUT_DIR):
#         if pdf_file.endswith(".pdf"):
#             process_pdf(os.path.join(INPUT_DIR, pdf_file))
    
#     end_time = time.time()  # End timer for the whole process
#     total_duration = end_time - start_time
#     logging.info(f"Total time for all PDF processing: {total_duration:.2f} seconds.")

# if __name__ == "__main__":
#     main()






import os
import logging
import csv
import pdfplumber
import shutil
import time
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Azure OpenAI credentials
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-mini"

# Initialize the Azure OpenAI model
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.1
)

# Directories setup
INPUT_DIR = "input"
OUTPUT_DIR = "output"
ARCHIVE_DIR = "archive"
DUMP_DIR = "dump"  # Directory for files with less than 50% data

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(DUMP_DIR, exist_ok=True)  # Create dump folder

# Configure logging to save only time-based logs
log_file_path = os.path.join(OUTPUT_DIR, "process_logs.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Display logs in the console
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8')  # Save logs to a text file
    ]
)

# Counter for auto-incrementing ID
id_counter = 0

def extract_pdf_text_and_tables(pdf_path):
    """Extract text and tables from a PDF."""
    full_text = ""
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw_text = page.extract_text(x_tolerance=2, y_tolerance=2)
            full_text += raw_text if raw_text else ""

            tables = page.extract_tables(
                table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"}
            )
            if tables:
                all_tables.extend(tables)

    return full_text, all_tables

def analyze_with_llm(raw_text, tables_text):
    prompt = (
        "You are an expert in extracting structured data from tool specification documents. "
        "Analyze the following text and extract key specifications for each tool model mentioned. "
        "For each tool model, extract the following details:\n"
        "- **Product_Usage**: Information about product usage\n"
        "- **Product_Sub_Category**: Information about Sub-category of product\n"
        "- **Product_Name**: Information about product name\n"
        "- **Product_Description**: Description of product\n"
        "- **Product_Configuration_or_Specifications**: Configuration/Specification of product\n"
        "- **Product_Variants**: Different Variants of the product\n"
        "- **Product_Colors**: Colors of the product\n"
        "- **Other_Features_or_attributes**: Features/Attributes of product\n"
        "- **Image_URLs**: Image links from the pro\n"
        "If a model is mentioned in a table or paragraph, ensure all following data is grouped under that model.\n\n"
        "For any of the above categories, if the information is presented in a table, ensure that the data is correctly associated with the corresponding model. No assumption, no approximate required.\n\n"
        f"Text extracted:\n{raw_text}\n\nTables:\n{tables_text}"
    )

    try:
        messages = [
            SystemMessage(content="You are a technical expert extracting structured data from tool specifications."),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)

        llm_output = response.content.strip()
        logging.debug(f"LLM Response:\n{llm_output}")  # Debugging, but not in the log file

        return llm_output
    except Exception as e:
        logging.error(f"Error calling Azure OpenAI API: {e}")
        return None

headers = ["Product_Usage", "id", "Product_Sub_Category", "Product_Name", "Product_Description", "Product_Configuration_or_Specifications", "Product_Variants", "Product_Colors", "Other_Features_or_attributes", "Image_URLs"]

def parse_llm_output(llm_output, raw_text, headers):
    global id_counter  # Use the global counter for auto-incrementing ID
    extracted_data = []
    models = llm_output.split("\n\n")  # Split by double newlines to separate each model's data
    total_columns = len(headers)  # Total columns dynamically based on headers

    for model in models:
        model_data = {}
        lines = model.split("\n")

        for line in lines:
            if ": " in line:
                key, value = line.split(": ", 1)
                key = key.strip().replace("**", "").strip()  # Remove markdown formatting
                value = value.strip()

                # Map the key to the correct column name
                if "Product_Usage" in key:
                    model_data["Product_Usage"] = value
                elif "Product_Sub_Category" in key:
                    model_data["Product_Sub_Category"] = value
                elif "Product_Name" in key:
                    model_data["Product_Name"] = value
                elif "Product_Description" in key:
                    model_data["Product_Description"] = value
                elif "Product_Configuration_or_Specifications" in key:
                    model_data["Product_Configuration_or_Specifications"] = value
                elif "Product_Variants" in key:
                    model_data["Product_Variants"] = value
                elif "Product_Colors" in key:
                    model_data["Product_Colors"] = value
                elif "Other_Features_or_attributes" in key:
                    model_data["Other_Features_or_attributes"] = value
                elif "Image_URLs" in key:
                    model_data["Image_URLs"] = value

        # Add auto-incrementing ID
        model_data["id"] = str(id_counter)
        id_counter += 1

        # Check if at least 50% of the columns have data
        filled_columns = sum(1 for key in model_data if model_data[key])
        if filled_columns >= total_columns * 0.5:  # At least 50% of columns have data
            extracted_data.append(model_data)

    return extracted_data

def save_as_csv(parsed_data, output_file_path):
    with open(output_file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)  # Ensure quoting is enabled for multiline fields
        writer.writerow(headers)

        for row in parsed_data:
            # Write row with proper formatting for multiline fields
            writer.writerow([ 
                row.get("Product_Usage", ""),
                row.get("id", ""),
                row.get("Product_Sub_Category", ""),
                row.get("Product_Name", ""),
                row.get("Product_Description", ""),
                row.get("Product_Configuration_or_Specifications", ""),
                row.get("Product_Variants", ""),
                row.get("Product_Colors", ""),
                row.get("Other_Features_or_attributes", ""),
                row.get("Image_URLs", ""),
            ])

    logging.info(f"CSV file saved: {output_file_path}")

def process_pdf(pdf_path):
    """Process a single PDF file."""
    start_time = time.time()  # Start timer for processing

    try:
        logging.info(f"Started processing PDF: {pdf_path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Start timer for reading file
        read_start_time = time.time()
        raw_text, tables = extract_pdf_text_and_tables(pdf_path)
        read_end_time = time.time()
        read_duration = read_end_time - read_start_time
        logging.info(f"Reading PDF took {read_duration:.2f} seconds.")
        
        tables_text = "\n".join(
            [" | ".join(map(str, row)) for table in tables for row in table]
        ) if tables else ""

        # Start timer for LLM invocation
        llm_start_time = time.time()
        analyzed_text = analyze_with_llm(raw_text, tables_text)
        llm_end_time = time.time()
        llm_duration = llm_end_time - llm_start_time
        logging.info(f"LLM invocation took {llm_duration:.2f} seconds.")
        
        if not analyzed_text:
            raise ValueError("No response from Azure OpenAI")

        structured_data = parse_llm_output(analyzed_text, raw_text, headers)
        if not structured_data:
            logging.warning(f"Skipping {pdf_path} as have data is less.")
            shutil.move(pdf_path, os.path.join(DUMP_DIR, os.path.basename(pdf_path)))  # Move to dump folder
            return

        # Start timer for writing CSV file
        csv_start_time = time.time()
        output_filename = os.path.join(OUTPUT_DIR, os.path.basename(pdf_path).replace(".pdf", "_output.csv"))
        save_as_csv(structured_data, output_filename)
        csv_end_time = time.time()
        csv_duration = csv_end_time - csv_start_time
        logging.info(f"Writing CSV file took {csv_duration:.2f} seconds.")
        
        shutil.move(pdf_path, os.path.join(ARCHIVE_DIR, os.path.basename(pdf_path)))
        logging.info(f"PDF processed and archived: {pdf_path}")

        end_time = time.time()  # End timer for processing
        total_duration = end_time - start_time
        logging.info(f"Total processing time for {pdf_path} was {total_duration:.2f} seconds.")

    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")

def main():
    """Main function to process all PDFs in the input directory."""
    start_time = time.time()  # Start timer for the whole process
    logging.info("Starting PDF processing...")

    for pdf_file in os.listdir(INPUT_DIR):
        if pdf_file.endswith(".pdf"):
            process_pdf(os.path.join(INPUT_DIR, pdf_file))
    
    end_time = time.time()  # End timer for the whole process
    total_duration = end_time - start_time
    logging.info(f"Total time for all PDF processing: {total_duration:.2f} seconds.")

if __name__ == "__main__":
    main()






