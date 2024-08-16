from dotenv import load_dotenv
import os 
import logging
import sys
import pandas as pd 
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

load_dotenv()

trade_weighted_avarage_indictive_rates = os.path.join("data", 'trade-weighted-average-indicative-rates.csv')

trade_weighted_avarage_indictive_rates_df  = pd.read_csv(trade_weighted_avarage_indictive_rates)


forex_query_engine = PandasQueryEngine(df=trade_weighted_avarage_indictive_rates_df,verbose=True, instruction_str=instruction_str)

forex_query_engine.update_prompts({"pandas_prompt": new_prompt})

forex_query_engine.query("What is the current rate of usd")

tools = [
    note_engine,
    QueryEngineTool(query_engine=forex_query_engine, metadata=ToolMetadata(
        name="forex_data",
        description="This provides information about the forex exchange rates published by the Central Bank of Kenya. These rates are intended to help those exchanging currencies gauge the value of the shilling on any given day."
    ))
]


llm = OpenAI(
    model="gpt-3.5-turbo-instruct3"
)

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while(prompt := input("enter a prompt (q to quit): ")) != "q": 
        result = agent.query(prompt)
        print(result)

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))