import argparse
import overpass
import json
import datetime
import logging
import os
import sys
from io import StringIO
from math import radians, sin, cos, sqrt, atan2
import requests  # Add requests for API calls
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
    PromptTemplate,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from playwright.sync_api import sync_playwright

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "georag_karlsruhe"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemma:2b"
DIMENSION = 384  # Dimension for all-MiniLM-L6-v2 embedding model
# Default user location, if no location is provided
# Setting Karlsruhe Palace for demo purposes
DEFAULT_USER_LAT = 49.0135
DEFAULT_USER_LON = 8.4044


# Overpass API setup
def get_osm_data(query: str) -> list:
    """
    Fetches data from OpenStreetMap using the Overpass API.

    Args:
        query (str): The Overpass QL query to execute.

    Returns:
        list: A list of features returned by the Overpass API.
    """
    api = overpass.API(timeout=180)
    try:
        response = api.get(query, responseformat="geojson")
        return response.get("features", [])
    except overpass.exceptions.OverpassGatewayTimeout:
        logger.error(
            "Overpass API Gateway Timeout: the query took too long to execute."
        )
        return []
    except Exception as e:
        logger.error(f"An error occurred while fetching data from Overpass API: {e}")
        return []


# Function to fetch data from OpenStreetMap & create documents
def create_poi_documents() -> list[Document]:
    """
    Creates documents from OpenStreetMap data for point(s) of interest in Karlsruhe.

    Returns:
        list: A list of Document objects containing POI information.
    """
    logger.info("Fetching OpenStreetMap data for Karlsruhe...")

    # Query for restaurants, cafes, attractions, & shopping centers in Karlsruhe
    query = """
    area[name="Karlsruhe"]->.searchArea;
    (
      node["amenity"~"restaurant|cafe"](area.searchArea);
      way["amenity"~"restaurant|cafe"](area.searchArea);
      node["tourism"~"attraction|theme_park|zoo|museum"](area.searchArea);
      way["tourism"~"attraction|theme_park|zoo|museum"](area.searchArea);
      node["shop"~"mall|department_store|supermarket|electronics|appliance"](area.searchArea);
      way["shop"~"mall|department_store|supermarket|electronics|appliance"](area.searchArea);
    );
    out center;
    """
    features = get_osm_data(query)

    # If no features are returned, log a warning & save an empty JSON
    if not features:
        logger.warning(
            "No features retrieved from OpenStreetMap, check your query or network."
        )
        with open("osm_features.json", "w", encoding="utf-8") as f:
            json.dump({"features": []}, f, indent=2, ensure_ascii=False)
        logger.info("Saved an empty feature list to osm_features.json.")
        return []

    # Output queried features to a JSON file for debugging/inspection
    with open("osm_features.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(features)} features to osm_features.json")

    # Create documents from the features
    documents = []
    # Descriptions for different types of shops
    shop_descriptions = {
        "mall": "a shopping center containing multiple stores.",
        "department_store": "a large store selling many varieties of goods, such as clothing, furniture, home goods, & large appliances like washing machines.",
        "supermarket": "a large store selling groceries & household goods like cleaning supplies & personal care items.",
        "electronics": "a shop that sells consumer electronics like TVs & computers. Larger ones may sell home appliances such as computers & fridges.",
        "appliance": "a shop that focuses in selling large electrical items such as washing machines, fridges, cookers, ovens, fans, etc.",
    }
    # Descriptions for different types of tourism spots
    tourism_descriptions = {
        "attraction": "an object of interest for a tourist, or a purpose-built tourist attraction.",
        "theme_park": "an amusement park with rides & attractions based on a specific theme.",
        "zoo": "a facility where animals are kept for public display & education.",
        "museum": "an institution that cares for a collection of artifacts & other objects of artistic, cultural, historical, or scientific importance.",
    }

    for feature in features:
        props = feature.get("properties", {})
        tags = props.get("tags", {})
        name = tags.get("name", "Unnamed")  # Default name

        # Extract coordinates from GeoJSON
        latitude, longitude = None, None
        if "geometry" in feature and "coordinates" in feature["geometry"]:
            lon, lat = feature["geometry"]["coordinates"]
            latitude, longitude = lat, lon  # Store

        description = f"This is a point of interest named '{name}, "

        amenity = tags.get("amenity")
        cuisine = tags.get("cuisine")
        shop = tags.get("shop")
        tourism = tags.get("tourism")

        if amenity and amenity in ["restaurant", "cafe"]:
            cuisine_info = f" that serves {cuisine} food" if cuisine else ""
            description += f"a {amenity}{cuisine_info}."
        if shop:
            description += shop_descriptions.get(shop, f"a shop of type '{shop}'.")
        if tourism:
            description += tourism_descriptions.get(
                tourism, f"a tourism spot of type '{tourism}'."
            )

        # Add other available tags for more context
        for key, value in tags.items():
            if key not in [
                "name",
                "amenity",
                "shop",
                "tourism",
                "cuisine",
                "addr:housenumber",
                "addr:street",
                "website",
                "phone",
            ]:
                description += f"{key.replace('_', ' ').title()}: {value}\n"

        # Add address information if available
        address_parts = []
        if tags.get("addr:street"):
            address_parts.append(tags["addr:street"])
        if tags.get("addr:housenumber"):
            address_parts.append(tags["addr:housenumber"])
        if address_parts:
            description += f"Address: {', '.join(address_parts)}\n"

        # Set metadata
        metadata = {
            "name": name,
            "id": feature.get("id"),
        }
        if shop:
            metadata["shop"] = shop
        if amenity:
            metadata["amenity"] = amenity
        if tourism:
            metadata["tourism"] = tourism
        if latitude is not None and longitude is not None:
            metadata["latitude"] = latitude
            metadata["longitude"] = longitude

        # Create Document object to store POI into vector store
        doc = Document(text=description.strip(), metadata=metadata)
        documents.append(doc)

    logger.info(f"Created {len(documents)} documents from OpenStreetMap data.")
    return documents


# Initialize Milvus vector store
def get_vector_store(overwrite: bool = False) -> MilvusVectorStore:
    """
    Initializes the Milvus vector store for storing POI documents.

    Args:
        overwrite (bool): Whether to overwrite the existing collection if it exists.

    Returns:
        MilvusVectorStore: An instance of the Milvus vector store.
    """
    logger.info(f"Connecting to Milvus vector store...")
    try:
        vector_store = MilvusVectorStore(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name=COLLECTION_NAME,
            dim=DIMENSION,  # Dimension of the embedding model
            overwrite=overwrite,
        )
        logger.info("Milvus vector store initialized successfully.")
        return vector_store
    except Exception as e:
        logger.error(
            f"Failed to connect to Milvus: {e}. Please ensure Milvus is running."
        )
        exit(1)  # Exit if Milvus connection fails


# Function to ingest data into the vector store
def ingest_data() -> None:
    """
    Ingests data from OpenStreetMap into the Milvus vector store.
    """
    logger.info("Starting data ingestion...")

    # Create documents from OpenStreetMap data
    documents = create_poi_documents()
    if not documents:
        logger.warning("No documents to ingest. Aborting ingestion.")
        return

    # Initialize the vector store, overwriting if specified
    vector_store = get_vector_store(overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Global settings for embedding model & LLM
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    # LLM is not needed for ingestion, set to None for efficiency
    Settings.llm = None
    # Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # Push documents into vector store
    logger.info(f"Creating index '{COLLECTION_NAME}' & ingesting documents...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,  # Show progress bar during ingestion
    )
    logger.info("Ingestion complete.")


def generate_maps_link(
    start_lat: float, start_lon: float, end_lat: float, end_lon: float
) -> str:
    """
    Generates a OpenStreetMap URL for directions between two points.

    Args:
        start_lat (float): Latitude of the starting point.
        start_lon (float): Longitude of the starting point.
        end_lat (float): Latitude of the destination point.
        end_lon (float): Longitude of the destination point.

    Returns:
        str: OpenStreetMap URL.
    """
    # Using the directions API format directly
    map_link = f"https://www.openstreetmap.org/directions?from={start_lat},{start_lon}&to={end_lat},{end_lon}"
    return map_link


def generate_static_map_url(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    output_dir: str,
    width: int = 800,
    height: int = 600,
) -> str:
    """
    Generates a static image of a route by taking a screenshot of an OpenStreetMap directions page using Playwright.

    Args:
        start_lat (float): Latitude of the starting point.
        start_lon (float): Longitude of the starting point.
        end_lat (float): Latitude of the destination point.
        end_lon (float): Longitude of the destination point.
        output_dir (str): The directory to save the screenshot in.
        width (int): Width of the screenshot image in pixels.
        height (int): Height of the screenshot image in pixels.

    Returns:
        str: File path of the generated screenshot.
    """
    osm_directions_url = f"https://www.openstreetmap.org/directions?from={start_lat},{start_lon}&to={end_lat},{end_lon}"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique filename for the screenshot
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"route_{timestamp}_{start_lat}_{start_lon}_to_{end_lat}_{end_lon}.png"
    filepath = os.path.join(output_dir, filename)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True
            )  # Use headless mode for automation
            page = browser.new_page()
            page.set_viewport_size({"width": width, "height": height})

            logger.info(
                f"Navigating to {osm_directions_url} to generate map screenshot..."
            )
            page.goto(osm_directions_url, wait_until="networkidle", timeout=60000)

            # Optional: Wait for a specific element that indicates the route is loaded, e.g., the route summary.
            # This makes the screenshot more reliable.
            try:
                page.wait_for_selector(".routing_summary", timeout=15000)
                logger.info("Route summary loaded, taking screenshot.")
            except Exception as e:
                logger.warning(
                    f"Route summary element not found, taking screenshot anyway. May be incomplete. Error: {e}"
                )

            page.screenshot(path=filepath)
            browser.close()
            logger.info(f"Screenshot saved to {filepath}")
            return filepath
    except Exception as e:
        logger.error(
            f"An error occurred while using Playwright to take a screenshot: {e}"
        )
        return ""  # Return empty string on failure


# Get route distance from OSRM API
def get_route_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Gets the driving distance between two points using the OSRM API.

    Args:
        lat1 (float): Latitude of the starting point.
        lon1 (float): Longitude of the starting point.
        lat2 (float): Latitude of the destination point.
        lon2 (float): Longitude of the destination point.

    Returns:
        float: The distance in kilometers, or infinity on error.
    """
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data["code"] == "Ok" and data["routes"]:
            distance_meters = data["routes"][0]["distance"]
            return distance_meters / 1000  # Convert to kilometers
        else:
            logger.warning(f"OSRM API could not find a route. Response: {data}")
            return float("inf")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling OSRM API: {e}")
        return float("inf")


# CLI command to submit query
def query_system(
    query_text: str,
    user_lat: float | None = None,
    user_lon: float | None = None,
    output_dir: str = "example_outputs",
) -> None:
    """
    Queries the RAG system with the provided query text. Automatically determines
    user location if not explicitly provided, by attempting to geocode from query.

    Args:
        query_text (str): User input to submit query to the RAG system.
        user_lat (float | None): User's current latitude. If None, use default location.
        user_lon (float | None): User's current longitude. If None, use default location.
        output_dir (str): Directory to save query outputs.
    """
    current_user_lat = user_lat
    current_user_lon = user_lon
    current_query_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Use default location if location is not provided
    if current_user_lat is None or current_user_lon is None:
        current_user_lat = DEFAULT_USER_LAT
        current_user_lon = DEFAULT_USER_LON

    logger.info(
        f"Processing query: '{query_text}' from location ({current_user_lat}, {current_user_lon})..."
    )

    vector_store = get_vector_store()

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=300)  # Increased timeout

    try:
        index = VectorStoreIndex.from_vector_store(vector_store)
    except Exception as e:
        logger.error(
            f"Failed to load index from vector store: {e}. Ensure data has been ingested."
        )
        return

    # Custom prompt template to force LLM to summarize context information & provide route
    qa_prompt_tmpl_str = (
        "You are an assistant that retrieves relevant information from the provided context about places in Karlsruhe, "
        "then summarizes it & recommend routes based on user query. "
        "Here is the context: \n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Do not repeat the user query in your reply. Keep it concise & helpful. "
        "Do not make up any information that is not in the context. "
        "Directly provide your reply with what you have summarized from the context information. "
        "If the query is asking for suggested places, start with sentence such as: "
        "'From your location at latitude {user_lat} & longitude: {user_lon}, these are the places that might interest you:'. "
        "If the query is double confirming certain information, start with sentence such as: "
        "'Based on the information available, here is what I found.'. "
        "Then provide relevant information that may help the user based on context information, such as operating hours, accessibility, and so on. "
        "At the end of your response, include this sentence: 'The route to the destination is attached below.' "
        "Are you ready? Here we go:"
        "\nUser Query: {query_str}\nYour reply:"
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    # Pass user_lat & user_lon directly into the query for the LLM to use
    # This ensures the LLM has access to the precise coordinates in its prompt.
    full_query_for_llm = f"{query_text}. My current location is latitude {current_user_lat} & longitude {current_user_lon}."

    # Initialize query_engine
    query_engine = index.as_query_engine(
        similarity_top_k=3,  # Retrieve top 3 most similar chunks
        text_qa_template=qa_prompt_tmpl,
    )

    response = query_engine.query(full_query_for_llm)

    # Sort source nodes by distance before passing to LLM to ensure top results (closest POIs) are prioritized
    source_nodes_with_distance = []
    for node in response.source_nodes:
        distance = float("inf")  # Default to infinity if no coords
        if "latitude" in node.node.metadata and "longitude" in node.node.metadata:
            poi_lat = node.node.metadata["latitude"]
            poi_lon = node.node.metadata["longitude"]
            distance = get_route_distance(
                current_user_lat, current_user_lon, poi_lat, poi_lon
            )
        source_nodes_with_distance.append({"node": node, "distance": distance})

    # Sort the list of nodes based on distance
    source_nodes_with_distance.sort(key=lambda x: x["distance"])

    # Re-assign the sorted nodes to the response object for consistent processing later
    sorted_source_nodes = [item["node"] for item in source_nodes_with_distance]
    response.source_nodes = sorted_source_nodes

    # Output the final prompt into LLM for debugging purposes
    context_str = "\n\n".join(
        [node.node.get_content() for node in response.source_nodes]
    )
    final_prompt = qa_prompt_tmpl.format(
        context_str=context_str,
        query_str=full_query_for_llm,  # Use the full query sent to LLM
        user_lat=current_user_lat,
        user_lon=current_user_lon,
    )

    # Save the full prompt to a file for debugging purposes
    # Create folder if folder is not available
    if not os.path.exists("llm_prompt_input"):
        os.makedirs("llm_prompt_input")
    prompt_filename = f"llm_prompt_input/llm_prompt_{current_query_time}.txt"
    with open(prompt_filename, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {current_query_time}\n\n")
        f.write(f"User Location: ({current_user_lat}, {current_user_lon})\n\n")
        f.write(f"Original Query: {query_text}\n\n")
        f.write(f"--- Full Prompt Sent to LLM ---\n{final_prompt}")
    logger.info(f"Full LLM prompt saved to {prompt_filename}")

    # Capture print output to a string
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # Output the LLM response
    print(
        "\n================================================================================="
    )
    print("----- LLM Response -----")
    print(response)
    print(
        "================================================================================="
    )

    print(
        "\n================================================================================="
    )
    print("----- Retrieved Sources -----")
    relevant_pois = []
    for i, node in enumerate(response.source_nodes):
        print(f"Source Node {i+1} ID: {node.node.id_}, Score: {node.score:.4f}")
        print(f"Content (first 200 chars): {node.node.get_content()}...")
        # Check if coordinates exist & add to relevant_pois
        if "latitude" in node.node.metadata and "longitude" in node.node.metadata:
            poi_name = node.node.metadata.get("name", "Unnamed POI")
            poi_lat = node.node.metadata["latitude"]
            poi_lon = node.node.metadata["longitude"]
            distance = get_route_distance(
                current_user_lat, current_user_lon, poi_lat, poi_lon
            )
            relevant_pois.append(
                {
                    "name": poi_name,
                    "latitude": poi_lat,
                    "longitude": poi_lon,
                    "distance_km": distance,
                    "node_content": node.node.get_content(),
                }
            )
        print("-----------------------")
    print(
        "================================================================================="
    )

    # Provide Google Maps links for relevant POIs with coordinates
    if relevant_pois:
        # Sort by distance to suggest the closest ones first
        relevant_pois.sort(key=lambda x: x["distance_km"])
        print(
            "================================================================================="
        )
        print("----- Recommended Routes (OpenStreetMaps) -----")
        for poi in relevant_pois[:3]:  # Show top 3 closest POIs
            maps_link = generate_maps_link(
                current_user_lat, current_user_lon, poi["latitude"], poi["longitude"]
            )
            print(f"Route to {poi['name']} (approx. {poi['distance_km']:.2f} km away):")
            print(f"  {maps_link}")
            print("-----------------------")
        print(
            "================================================================================="
        )
    else:
        print("\nNo specific points of interest found in retrieved context.")

    console_output = mystdout.getvalue()  # Capture the output
    sys.stdout = old_stdout  # Restore stdout
    print(console_output)  # Print captured output to console

    # Save output into markdown files
    markdown_output = f"# Query Result - {current_query_time}\n\n"
    markdown_output += f"**Original Query:** `{query_text}`\n\n"
    markdown_output += "## LLM Response\n\n"
    markdown_output += f"```\n{response}\n```\n\n"  # Use the direct response

    markdown_output += "## Retrieved Sources\n\n"
    for i, node in enumerate(response.source_nodes):
        markdown_output += (
            f"### Source Node {i+1} (ID: {node.node.id_}, Score: {node.score:.4f})\n"
        )
        markdown_output += f"```\n{node.node.get_content()}\n```\n\n"

    # Provide Google Maps links & static map images for relevant POIs
    if relevant_pois:
        relevant_pois.sort(key=lambda x: x["distance_km"])
        markdown_output += "## Recommended Routes\n\n"
        for poi in relevant_pois[:5]:
            maps_link = generate_maps_link(
                current_user_lat, current_user_lon, poi["latitude"], poi["longitude"]
            )
            # Generate the screenshot, which now saves a local file
            screenshot_path = generate_static_map_url(
                current_user_lat,
                current_user_lon,
                poi["latitude"],
                poi["longitude"],
                output_dir=output_dir,
            )

            markdown_output += f"### Route to {poi['name']} (approx. {poi['distance_km']:.2f} km away)\n"
            markdown_output += f"**[View on OpenStreetMaps]({maps_link})**\n\n"

            if screenshot_path:
                # Use a relative path for the markdown image link
                image_filename = os.path.basename(screenshot_path)
                markdown_output += f"![Route to {poi['name']}]({image_filename})\n\n"
            else:
                markdown_output += "*Map image could not be generated.*\n\n"

    # Export finalized markdown output
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, f"query_{current_query_time}.md")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown_output)
    logger.info(f"Output for query saved to {output_filename}")


def run_example_queries(
    user_lat: float = DEFAULT_USER_LAT,
    user_lon: float = DEFAULT_USER_LON,
    output_dir: str = "example_outputs",
) -> None:
    """
    Runs predefined example queries & saves their outputs to files.

    Args:
        user_lat (float): User's current latitude for example queries.
        user_lon (float): User's current longitude for example queries.
        output_dir (str): Directory to save example query outputs.
    """
    example_queries = [
        "I am looking for a good sushi restaurant near Europaplatz.",
        "Where can I bring my 2 kids to visit in Karlsruhe?",
        "I want to buy a new washing machine, where can I go?",
        "What are some good places to eat near here?",
        "Does ON Sushi have vegetarian options?",
    ]

    for i, query_text in enumerate(example_queries):
        logger.info(f"\n--- Running Example Query {i+1}: {query_text} ---")
        # Call query_system, which will handle its own output saving
        query_system(
            query_text, user_lat=user_lat, user_lon=user_lon, output_dir=output_dir
        )
        logger.info(f"--- Finished Example Query {i+1} ---")


# Command-line interface here
def main() -> None:
    """
    Main function to handle command-line arguments & execute the appropriate command.
    """

    parser = argparse.ArgumentParser(description="GeoRAG system for Karlsruhe POIs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    parser_ingest = subparsers.add_parser(
        "ingest", help="Ingest data from OpenStreetMap into Milvus."
    )

    # Query command
    parser_query = subparsers.add_parser("query", help="Query the RAG system.")
    parser_query.add_argument("query_text", type=str, help="The question to ask.")
    # Made user_lat/lon optional
    parser_query.add_argument(
        "--user_lat",
        type=float,
        default=None,
        help=f"User's current latitude (optional, will use default location if not provided).",
    )
    parser_query.add_argument(
        "--user_lon",
        type=float,
        default=None,
        help=f"User's current longitude (optional, will use default location if not provided).",
    )
    parser_query.add_argument(
        "--output_dir",
        type=str,
        default="example_outputs",
        help="Directory to save example query outputs (default: example_outputs).",
    )

    # Example command
    parser_example = subparsers.add_parser(
        "example", help="Run predefined example queries & save outputs."
    )
    parser_example.add_argument(
        "--user_lat",
        type=float,
        default=None,
        help=f"User's current latitude (optional, will use default location if not provided).",
    )
    parser_example.add_argument(
        "--user_lon",
        type=float,
        default=None,
        help=f"User's current longitude (optional, will use default location if not provided).",
    )
    parser_example.add_argument(
        "--output_dir",
        type=str,
        default="example_outputs",
        help="Directory to save example query outputs (default: example_outputs).",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_data()
    elif args.command == "query":
        query_system(args.query_text, args.user_lat, args.user_lon, args.output_dir)
    elif args.command == "example":
        run_example_queries(args.user_lat, args.user_lon, args.output_dir)


# Main entry point
if __name__ == "__main__":
    main()
