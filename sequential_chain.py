from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# create a prompt 
prompt1=PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate a 5 pointer summery from the following text \n {test}',
    input_variables=['text']
)

# use parser for structure output
parser=StrOutputParser()

# pipeline
chain=prompt1 |model |parser |prompt2 |model |parser

result=chain.invoke({'topic':'unemployment in india '})

print(result)

# visualize the chain 
chain.get_graph().print_ascii()