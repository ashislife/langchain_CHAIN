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

prompt=PromptTemplate(
    template='Generate 5 interesting Fact about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

# pipeline (prompt->model ->parser)
chain=prompt |model |parser
result=chain.invoke({'topic':'Cricket'})

print(result)

# visualize the chain 
chain.get_graph().print_ascii()