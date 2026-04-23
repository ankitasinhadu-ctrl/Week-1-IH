"""
Product Listing Generator
==========================
A script to generate product listings using OpenAI's ChatGPT Vision API.

Usage:
    python product_listing_generator.py

Requirements:
    - OpenAI API key (set in .env file as OPENAI_API_KEY)
    - datasets library (pip install datasets)
"""

import os
import json
import base64
import io
import time
from pathlib import Path

# Third-party imports
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
from PIL import Image
import pandas as pd


# ============================================
# CONFIGURATION
# ============================================

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================
# FUNCTIONS
# ============================================

def encode_image_to_base64(image_data):
    """
    Convert image data to base64 string.
    
    Parameters:
    - image_data: PIL Image or bytes
    
    Returns:
    - Base64 encoded string or None on error
    """
    try:
        if isinstance(image_data, bytes):
            return base64.b64encode(image_data).decode('utf-8')
        elif isinstance(image_data, Image.Image):
            buffer = io.BytesIO()
            image_data.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def create_product_listing_prompt(product_name, price, category, additional_info=None):
    """
    Create a prompt for generating product listings.
    
    Parameters:
    - product_name: Name of the product
    - price: Price of the product
    - category: Product category
    - additional_info: Optional additional information
    
    Returns:
    - Formatted prompt string
    """
    prompt = f"""You are an expert e-commerce copywriter. Analyze the product image and create a compelling product listing.
 
Product Information:
- Name: {product_name}
- Price: ${price:.2f}
- Category: {category}
{f'- Additional Info: {additional_info}' if additional_info else ''}
 
Please create a professional product listing that includes:
 
1. **Product Title** (catchy, SEO-friendly, 60 characters max)
2. **Product Description** (detailed, 150-200 words)
   - Highlight key features and benefits
   - Use persuasive language
   - Include relevant details visible in the image
3. **Key Features** (bullet points, 5-7 items)
4. **SEO Keywords** (comma-separated, 10-15 relevant keywords)
 
Format your response as JSON with the following structure:
{{
    "title": "Product title here",
    "description": "Full description here",
    "features": ["Feature 1", "Feature 2", ...],
    "keywords": "keyword1, keyword2, ..."
}}
 
Be specific about what you see in the image. Mention colors, materials, design elements, and any distinctive features."""
    
    return prompt


def call_chatgpt_vision(image_data, product_name, price, category, additional_info=None):
    """
    Call ChatGPT API with product image and information.
    
    Parameters:
    - image_data: PIL Image or bytes
    - product_name: Name of the product
    - price: Product price
    - category: Product category
    - additional_info: Optional additional info
    
    Returns:
    - Parsed JSON response or error message
    """
    # Encode image to base64
    encoded_image = encode_image_to_base64(image_data)
    if not encoded_image:
        return {"error": "Failed to encode image"}
    
    # Create prompt
    prompt = create_product_listing_prompt(product_name, price, category, additional_info)
    
    try:
        # Call OpenAI API with vision
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        # Extract response content
        content = response.choices[0].message.content
        
        # Parse JSON from response
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "No JSON found in response", "raw": content}
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "raw": content}
            
    except Exception as e:
        return {"error": str(e)}


def load_product_dataset(num_products=100):
    """
    Load product dataset from HuggingFace.
    
    Parameters:
    - num_products: Number of products to load
    
    Returns:
    - DataFrame with product data
    """
    print(f"Loading {num_products} products from HuggingFace...")
    dataset = load_dataset("ashraq/fashion-product-images-small", split=f"train[:{num_products}]")
    products_df = pd.DataFrame(dataset)
    print(f"✓ Loaded {len(products_df)} products")
    return products_df


def process_multiple_products(products_df, start_idx=0, num_products=3, sample_price=49.99):
    """
    Process multiple products and generate listings.
    
    Parameters:
    - products_df: DataFrame with product data
    - start_idx: Starting index
    - num_products: Number of products to process
    - sample_price: Sample price to use
    
    Returns:
    - Tuple of (results, errors)
    """
    results = []
    errors = []
    
    print(f"\nProcessing {num_products} products starting from index {start_idx}...")
    print("="*50)
    
    for i in range(start_idx, min(start_idx + num_products, len(products_df))):
        product = products_df.iloc[i]
        product_name = product['productDisplayName']
        category = product['masterCategory']
        image = product['image']
        
        print(f"\n[{i - start_idx + 1}/{num_products}] Processing: {product_name[:40]}...")
        
        try:
            result = call_chatgpt_vision(
                image_data=image,
                product_name=product_name,
                price=sample_price,
                category=category
            )
            
            if "error" not in result:
                results.append({
                    "index": i,
                    "product_name": product_name,
                    "success": True,
                    "data": result
                })
                print(f"  ✓ Success")
            else:
                errors.append({
                    "index": i,
                    "product_name": product_name,
                    "error": result['error']
                })
                print(f"  ✗ Error: {result['error']}")
                
        except Exception as e:
            errors.append({
                "index": i,
                "product_name": product_name,
                "error": str(e)
            })
            print(f"  ✗ Exception: {str(e)[:50]}")
        
        # Rate limiting
        if i < start_idx + num_products - 1:
            time.sleep(1)
    
    return results, errors


def save_results(results, errors, output_file="generated_listings.json"):
    """
    Save results to JSON file.
    
    Parameters:
    - results: List of successful results
    - errors: List of errors
    - output_file: Output filename
    
    Returns:
    - None
    """
    output_data = {
        "summary": {
            "total_processed": len(results),
            "successful": len(results),
            "errors": len(errors)
        },
        "listings": results,
        "errors": errors
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {output_file}")
    return output_data


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("PRODUCT LISTING GENERATOR")
    print("="*60)
    
    # Step 1: Load dataset
    products_df = load_product_dataset(num_products=100)
    
    # Step 2: Process multiple products
    all_results, all_errors = process_multiple_products(
        products_df, 
        start_idx=0, 
        num_products=3,
        sample_price=49.99
    )
    
    # Step 3: Save results
    output_data = save_results(all_results, all_errors)
    
    # Step 4: Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total processed: {len(all_results)}")
    print(f"Successful: {len(all_results)}")
    print(f"Errors: {len(all_errors)}")
    
    for r in all_results:
        title = r['data'].get('title', 'N/A')[:50]
        print(f"  • {r['product_name'][:30]}... → {title}")