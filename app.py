import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain import PromptTemplate, FewShotPromptTemplate, LLMChain

import os
import json

from dotenv import load_dotenv
load_dotenv(".env", override=True)

def load_pretrain_model():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    return llm

def load_gpt4_local():
    pass

def create_machine_template(llm):
    template = """You are a programmer, your job is to generate source code with html, css and javascript to solve the problem with description below.\
    If the question cannot be answered using the information provided, answer with "I don't know".
    ...
    Question: {query}
    Answer: """

    prompt_template = PromptTemplate(
                        input_variables=["query"],
                        template=template
                    )
    
    # Create the LLMChain for the prompt
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    return chain

def create_machine_template_(llm):
    examples = [
    {
        "query": "Create the structure for a product listing page with placeholders for product images, titles, prices, and 'Add to Cart' buttons.",
        "answer": """HTML:\
<!DOCTYPE html>\
<html lang="en">\
<head>\
<meta charset="UTF-8">\
<meta name="viewport" content="width=device-width, initial-scale=1.0">\
<title>Product Listing Page</title>\
<link rel="stylesheet" href="styles.css">\
</head>\
<body>\
<div class="container">\
<div class="product">\
<img src="placeholder_image.jpg" alt="Product Image">\
<h2 class="title">Product Title</h2>\
<p class="price">$XX.XX</p>\
<button class="add-to-cart-btn">Add to Cart</button>\
</div>\
<!-- Repeat this structure for each product -->\
</div>\
</body>\
</html>
CSS (styles.css):\
.container {\
display: flex;
flex-wrap: wrap;
justify-content: space-around;
}
.product {\
width: 200px;
margin: 20px;
text-align: center;\
}
.product img {\
width: 100%;
max-width: 200px;
height: auto;\
}
.title {\
margin-top: 10px;\
}
.price {\
font-weight: bold;\
}
.add-to-cart-btn {\
margin-top: 10px;
background-color: #007bff;
color: #fff;
border: none;
padding: 10px 20px;
border-radius: 5px;
cursor: pointer;
transition: background-color 0.3s;\
}
.add-to-cart-btn:hover {\
background-color: #0056b3;\
} """
    }, 
    {
        "query": "Design the layout for a product details page with sections for displaying product images, descriptions, specifications, and customer reviews.",
        "answer": """HTML:\
<!DOCTYPE html>\
<html lang="en">\
<head>\
<meta charset="UTF-8">\
<meta name="viewport" content="width=device-width, initial-scale=1.0">\
<title>Product Details</title>\
<link rel="stylesheet" href="styles.css">\
</head>\
<body>\
<div class="container">\
<div class="product-images">\
<img src="product_image1.jpg" alt="Product Image 1">\
<img src="product_image2.jpg" alt="Product Image 2">\
<img src="product_image3.jpg" alt="Product Image 3">\
</div>\
<div class="product-details">\
<h1 class="title">Product Title</h1>\
<p class="description">\
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam gravida magna quis purus vehicula, eu pharetra sem finibus. Fusce sollicitudin orci non magna vestibulum, a pharetra sem luctus.\
</p>\
<h2 class="specifications-title">Specifications</h2>\
<ul class="specifications">\
<li>Size: </li>\
<li>Color: </li>\
<li>Material: </li>\
<!-- Add more specifications here -->\
</ul>\
</div>\
<div class="customer-reviews">\
<h2 class="reviews-title">Customer Reviews</h2>\
<div class="review">\
<div class="reviewer">John Doe</div>\
<div class="rating">★★★★☆</div>\
<p class="review-text">\
"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer nec odio. Praesent libero. Sed cursus ante dapibus diam."\
</p>\
</div>\
<!-- Add more reviews here -->\
</div>\
</div>\
</body>\
</html>\
CSS (styles.css):
.container {\
max-width: 800px;
margin: 0 auto;\
}
.product-images {\
display: flex;
justify-content: space-between;
margin-bottom: 20px;\
}
.product-images img {\
width: 30%;
max-height: 200px;\
}
.product-details {\
margin-bottom: 20px;\
}
.title {\
margin-bottom: 10px;\
}
.specifications-title {\
margin-bottom: 5px;\
}
.specifications {\
list-style: none;
padding: 0;\
}
.customer-reviews {\
border-top: 1px solid #ccc;
padding-top: 20px;\
}
.reviews-title {\
margin-bottom: 10px;\
}
.review {\
margin-bottom: 20px;\
}
.reviewer {\
font-weight: bold;\
}
.rating {\
color: #ffd700; /* gold */\
}
.review-text {\
margin-top: 5px;\
}"""
    }
]
    
    example_template = """
    User: {query}
    AI: {answer}
    """

    example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
    )
    
    prefix = """The following are excerpts from conversations with an AI
    life coach. The assistant provides insightful and practical advice to the users' questions. Here are some
    examples: 
    """

    suffix = """
    User: {query}
    AI: """

    few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
    )
    # Create the LLMChain for the few-shot prompt template
    chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)
    return chain



def generate_response(st, chain, input_data):
    # import ipdb; ipdb.set_trace(context=10)
    response = chain.run(input_data)
    st.info(response)

if __name__ == "__main__":
    st.title('🦜🔗 Quickstart App')
    openai_api_key = os.getenv("OPENAI_API_KEY")

    llm = load_pretrain_model()
    machine_chain = create_machine_template(llm)

    with st.form('my_form'):
        text = st.text_area('Enter text:', 'What problem do you want to handle?')
        submitted = st.form_submit_button('Submit')
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and openai_api_key.startswith('sk-'):
            generate_response(st=st,
                            chain=machine_chain,
                              input_data={"query": text})