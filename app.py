import os
import logging
import csv
import time
import io
import random
from dotenv import load_dotenv
from PIL import Image
import fitz
import camelot
from azure.storage.blob import BlobServiceClient
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.settings import Settings

# Load environment variables
load_dotenv()

# Set up Azure OpenAI credentials
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = "sbd-gpt-4o"
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME = "sbd-text-embedding-ada-002"

# Initialize the Azure OpenAI model
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.1
)

# Configure LlamaIndex to use Azure OpenAI embeddings
Settings.embed_model = AzureOpenAIEmbedding(
    deployment_name=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# Configure LlamaIndex to use Azure OpenAI
Settings.llm = AzureOpenAI(
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.1
)

# Azure Blob Storage setup
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_INPUT_CONTAINER = "data-mactool-user-manuals"
BLOB_PROCESSED_CONTAINER = "data-mactool-user-manuals-processed"
BLOB_IMAGES_CONTAINER = "data-mactool-user-manuals-images"

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
input_container_client = blob_service_client.get_container_client(BLOB_INPUT_CONTAINER)
processed_container_client = blob_service_client.get_container_client(BLOB_PROCESSED_CONTAINER)
images_container_client = blob_service_client.get_container_client(BLOB_IMAGES_CONTAINER)

# Initializing temperory directory
temp_dir = os.getenv("TEMP_DIR")

# Ensure the directory exists
os.makedirs(temp_dir, exist_ok=True)

# Configure logging to log to both console and a file
log_file_path = os.path.join(temp_dir, "process_logs.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    ]
)

HEADERS = [
    "Product_Usage", "id", "Product_Sub_Category", "Product_Name", "Product_Description",
    "Product_Configuration_or_Specifications", "Product_Variants", "Product_Colors",
    "Other_Features_or_attributes", "Image_URLs"
]

def extract_pdf_text(pdf_path):
    """Use LlamaIndex Recursive Retriever to extract structured text from PDFs."""
    logging.info(f"Started text extraction for {pdf_path}")
    start_time = time.time()
    try:
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Create a retriever dictionary for RecursiveRetriever
        retriever_dict = {
            "root": index.as_retriever(similarity_top_k=5)
        }

        retriever = RecursiveRetriever(
            retriever_dict=retriever_dict,
            root_id="root"
        )

        query_engine = RetrieverQueryEngine(retriever=retriever)
        extracted_text = query_engine.query("Extract all relevant text content.")
        duration = time.time() - start_time
        logging.info(f"Completed text extraction for {pdf_path} in {duration:.2f} seconds")
        return extracted_text.response.strip() if extracted_text.response else ""
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_pdf_tables(pdf_path):
    """Use Camelot to extract tables from PDFs."""
    logging.info(f"Started table extraction for {pdf_path}")
    start_time = time.time()
    tables_text = []
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        if tables.n > 0:
            for table in tables:
                tables_text.append(table.df.to_string(index=False, header=False))
        else:
            logging.warning(f"No tables found in {pdf_path}")
    except Exception as e:
        logging.error(f"Error extracting tables from {pdf_path}: {e}")
    duration = time.time() - start_time
    logging.info(f"Completed table extraction for {pdf_path} in {duration:.2f} seconds")
    return "\n\n".join(tables_text) if tables_text else ""

def extract_images_from_pdf(pdf_path):
    """Extract images from a PDF and convert them to JPEG."""
    logging.info(f"Started image extraction for {pdf_path}")
    start_time = time.time()
    image_files = []
    try:
        doc = fitz.open(pdf_path)
        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            if not image_list:
                continue
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    image_pil = Image.open(io.BytesIO(image_bytes))
                    if image_pil.mode not in ["RGB"]:
                        image_pil = image_pil.convert("RGB")
                except Exception as e:
                    logging.error(f"Error processing image on page {page_index+1}, image {img_index+1}: {e}")
                    continue
                image_filename = f"{os.path.basename(pdf_path).replace('.pdf', '')}_page{page_index+1}_img{img_index+1}.jpeg"
                image_buffer = io.BytesIO()
                image_pil.save(image_buffer, format="JPEG")
                image_buffer.seek(0)
                blob_client = images_container_client.get_blob_client(f"Product/{image_filename}")
                blob_client.upload_blob(image_buffer, overwrite=True)
                image_files.append(image_filename)
        doc.close()
    except Exception as e:
        logging.error(f"Error extracting images from PDF {pdf_path}: {e}")
    duration = time.time() - start_time
    logging.info(f"Completed image extraction for {pdf_path} in {duration:.2f} seconds")
    return image_files

def analyze_with_llm(raw_text, tables_text):
    """Send extracted text & tables to Azure OpenAI for analysis."""
    logging.info("Started LLM invocation for analysis")
    start_time = time.time()
    if not raw_text and not tables_text:
        logging.warning("No content to analyze. Skipping LLM call.")
        return None

    prompt = (
        "You are an expert in extracting structured data from tool specification documents. "
        "Analyze the following text and extract key specifications for each tool model mentioned. "
        "For each tool model, extract the following details:\n"
        "- Product_Usage\n"
        "- Product_Sub_Category\n"
        "- Product_Name\n"
        "- Product_Description (Include any introductory or concluding remarks about the tool model here, along with any general descriptions.)\n"
        "- Product_Configuration_or_Specifications (If available include the specifications that are given in comma seperated format (eg. dimension: [value], max torque: [value], weight: [value], free speed: [value] etc.))\n"
        "- Product_Variants\n"
        "- Product_Colors\n"
        "- Other_Features_or_attributes\n"
        "- Image_URLs\n"
        "Format each entry as:\n"
        "**Product_Usage**: [value]\n"
        "**Product_Sub_Category**: [value]\n"
        "**Product_Name**: [value]\n"
        "**Product_Description**: [value]\n"
        "**Product_Configuration_or_Specifications**: [value]\n"
        "**Product_Variants**: [value]\n"
        "**Product_Colors**: [value]\n"
        "**Other_Features_or_attributes**: [value]\n"
        "**Image_URLs**: [value]\n\n"
        f"Text extracted:\n{raw_text}\n\nTables:\n{tables_text}"
    )
    try:
        messages = [
            SystemMessage(content="You are a technical expert extracting structured data from tool specifications."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        duration = time.time() - start_time
        logging.info(f"LLM invocation completed in {duration:.2f} seconds")
        return response.content.strip() if response.content else None
    except Exception as e:
        logging.error(f"Error calling Azure OpenAI API: {e}")
        return None

def parse_llm_output(llm_output, image_files):
    """Parses the structured response from LLM and formats it for CSV output."""
    extracted_data = []
    models = llm_output.strip().split("\n\n") if llm_output else []
    total_columns = len(HEADERS)

    try:
        for model in models:
            model_data = {header: "" for header in HEADERS}
            lines = model.split("\n")
            full_description = ""
            extracted_fields = {}

            for line in lines:
                if ": " in line:
                    key, value = line.split(": ", 1)
                    key = key.strip().replace("**", "").strip()
                    value = value.strip()
                    extracted_fields[key] = value
                else:
                    full_description += line + "\n"

            for key, value in extracted_fields.items():
                if "Product_Usage" in key:
                    model_data["Product_Usage"] = value
                elif "Product_Sub_Category" in key:
                    model_data["Product_Sub_Category"] = value
                elif "Product_Name" in key:
                    model_data["Product_Name"] = value
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

            description_parts = []
            if full_description:
                description_parts.append(full_description.strip())
            if "Product_Description" in extracted_fields:
                description_parts.append(extracted_fields["Product_Description"])
            model_data["Product_Description"] = "\n".join(description_parts).strip()

            model_data["id"] = str(random.randint(1000, 9999))

            if image_files:
                model_data["Image_URLs"] = ", ".join(image_files)

            filled_columns = sum(1 for v in model_data.values() if v)
            if filled_columns >= total_columns * 0.5:
                extracted_data.append(model_data)
    except Exception as e:
        logging.error(f"Error while parsing LLM output: {e}")

    return extracted_data

def save_as_csv(parsed_data, output_blob_client):
    """Saves the extracted data into a CSV file with the specified headers."""
    logging.info("Started writing CSV file")
    start_time = time.time()
    try:
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADERS)
        for row in parsed_data:
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
        output_blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)
        duration = time.time() - start_time
        logging.info(f"CSV file written and uploaded as {output_blob_client.blob_name} in {duration:.2f} seconds")
    except Exception as e:
        logging.error(f"Error while saving CSV file: {e}")

def process_pdf(pdf_blob):
    """Process a single PDF file."""
    file_start_time = time.time()
    logging.info(f"Started processing PDF: {pdf_blob.name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        blob_client = input_container_client.get_blob_client(pdf_blob)
        # Download the PDF blob to a temporary file
        temp_pdf_path = os.path.join(temp_dir, os.path.basename(pdf_blob.name))
        download_start = time.time()
        with open(temp_pdf_path, "wb") as temp_pdf_file:
            temp_pdf_file.write(blob_client.download_blob().readall())
        download_duration = time.time() - download_start
        logging.info(f"Downloaded {pdf_blob.name} in {download_duration:.2f} seconds")

        # Extract text, tables, and images
        raw_text = extract_pdf_text(temp_pdf_path)
        tables_text = extract_pdf_tables(temp_pdf_path)
        image_files = extract_images_from_pdf(temp_pdf_path)
        if image_files:
            logging.info(f"Images extracted for {pdf_blob.name}: {', '.join(image_files)}")

        # Analyze with LLM
        llm_start = time.time()
        analyzed_text = analyze_with_llm(raw_text, tables_text)
        llm_duration = time.time() - llm_start
        logging.info(f"LLM processing took {llm_duration:.2f} seconds")
        if not analyzed_text:
            raise ValueError("No response from Azure OpenAI")

        # Parse LLM output and generate CSV
        structured_data = parse_llm_output(analyzed_text, image_files)
        if not structured_data:
            logging.warning(f"Skipping {pdf_blob.name} as insufficient data was extracted.")
            # Move the blob to the Error folder
            error_blob_client = input_container_client.get_blob_client(f"Product/Error/{os.path.basename(pdf_blob.name)}")
            error_blob_client.start_copy_from_url(blob_client.url)
            blob_client.delete_blob()
            return

        output_filename = os.path.basename(pdf_blob.name).replace(".pdf", "_output.csv")
        output_blob_client = processed_container_client.get_blob_client(f"Product/{output_filename}")
        save_as_csv(structured_data, output_blob_client)

        # Move the blob to the Success folder
        success_blob_client = input_container_client.get_blob_client(f"Product/Success/{os.path.basename(pdf_blob.name)}")
        success_blob_client.start_copy_from_url(blob_client.url)
        blob_client.delete_blob()
        logging.info(f"PDF processed and archived: {pdf_blob.name}")

        file_duration = time.time() - file_start_time
        logging.info(f"Total processing time for {pdf_blob.name}: {file_duration:.2f} seconds")
    except Exception as e:
        logging.error(f"Error processing {pdf_blob.name}: {e}")

def main():
    """Main function to process the first 5 PDFs in the input directory."""
    try:
        start_time = time.time()  # Start timer for the whole process
        logging.info("Starting PDF processing...")

        # Counter to limit processing to the first 5 files
        pdf_count = 0
        for pdf_blob in input_container_client.list_blobs(name_starts_with="Product/"):
            if pdf_blob.name.endswith(".pdf"):
                process_pdf(pdf_blob)
                pdf_count += 1
                if pdf_count >= 5:  # Stop after processing 5 files
                    break

        end_time = time.time()  # End timer for the whole process
        total_duration = end_time - start_time
        logging.info(f"Total time for processing the first 5 PDFs: {total_duration:.2f} seconds.")

    except Exception as e:
        logging.error(f"Error while processing PDFs: {e}")

if __name__ == "__main__":
    main()