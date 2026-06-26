from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# pydantic for always same output 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel ,Field
from typing import Literal
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda

load_dotenv()

# define the model 
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# create a parser for string structured output 
parser=StrOutputParser()


# create a pydantic class for structure or same result for all time 

class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description='Give the sentiment of the feedback')


# create a pydantic parser 
parser2=PydanticOutputParser(pydantic_object=Feedback)

# create a prompt 
prompt1=PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}

)


# pipeline 
classifier_chain=prompt1 |model |parser2


# result=classifier_chain.invoke({'feedback':'This is a terrible smartphone'}).sentiment

# print(result)


# now branching (conditional chain) start and before classificaiton done 


prompt2=PromptTemplate(
    template='Write a approprate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='Write a approprate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2 |model |parser),
    (lambda x:x.sentiment=='negative',prompt3 |model |parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain=classifier_chain |branch_chain

print(chain.invoke({'feedback':'This is a terrible phone'}))

