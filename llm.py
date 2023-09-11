from keys import mykey, serpkey

import os

from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = mykey
os.environ["SERPAPI_API_KEY"] = serpkey

from langchain.utilities import SerpAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.agents import AgentType, initialize_agent, load_tools, Tool

llm = OpenAI(temperature=0.6)


params = {
    "engine": "bing",
    "gl": "us",
    "location": "Milpitas, California, United States",
    "hl": "en",
}
search = SerpAPIWrapper(params=params)

serptool = Tool(
    name="Serp_location",
    description="chains from the search and location hopefully",
    func=search.run,
)

# a = (search.run("What are good coffee places near me?"))

# prompt_template_search = PromptTemplate(
#     input_variables= ['search'],
#     template="I want you to go through this {search} and list out the names of the coffee places that are in it"
# )



# tools = load_tools(["serpapi", 'llm-math'], llm= llm)
tools = [
    Tool(
    name="Serp_location",
    description="chains from the search and location hopefully",
    func=search.run,
)
]
agent  = initialize_agent(tools,llm, agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)

print(agent.run("What are good Chinese Places near me? Just tell me the names"))




prompt_template_name = PromptTemplate(
    input_variables= ['cuisine'],
    template="I want to open a restuarant for {cuisine} food. Suggest a fancy name for it"
)

name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

print(name_chain.run("Indian"))
# prompt_template_items = PromptTemplate(
#     input_variables= ['restaurant_name'],
#     template="Suggest some menu items for {restaurant_name}. Return it as a comma seperated list"
# )



# prompt_template_name.format(cuisine= "Italian")

# name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
# food_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key= "menu_items")

# chain = SequentialChain(
#     chains= [name_chain, food_chain],
#     input_variables=['cuisine'],
#     output_variables=['restaurant_name', "menu_items"]
# )

# print(chain({'cuisine': 'Marathi'}))