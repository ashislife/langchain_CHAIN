from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# for paralllel chain (parallely execute parallel chain)
from langchain.schema.runnable import RunnableParallel

load_dotenv()

# model 1
llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm1)

# model2

llm2 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)

model2 = ChatHuggingFace(llm=llm2)

# Two sequential prompt
prompt1=PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='Generate 5 short question answer from the following text \n {text}',
    input_variables=['text']
)

# One final prompt for merge both output 
prompt3=PromptTemplate(
    template='Merge the provided notes and quiz into a single document  \n notes -> {notes} and quiz ->{quiz}',
    input_variables=['notes','quiz']
)

# generate parser for string output with structured formate 
parser=StrOutputParser()

# parallel chain 
parallel_chain=RunnableParallel({
    'notes':prompt1 |model1 |parser ,
    'quiz':prompt2 |model2| parser

})

# merge parellel chain 
merge_chain=prompt3 |model1| parser

# final chain (treate as a sequential chain)
chain =parallel_chain |merge_chain

# text 
text="""SVM was first proposed by Vapnik in 1995 [1] and it
was named as “support-vector networks” in the early
time, which is a binary classification algorithm and implements the idea of mapping non-linearly vectors to a
very high-dimension feature space to construct a linear
decision surface (hyperplane) in this feature space. In
hence, it can achieve good effect on solving separable
and non-separable problems. This hyperplane is optimal in the sense of being a maximal margin classier
with respect to the training data [2]. The Structural
Risk Minimisation (SRM) principle, SVM follows, equips SVM with a greater ability to generalise. Corresponds to such significant advantages, SVM was applied to
classification and regression problems quickly [3]. Even
more, SVM embodies the characteristics of small samples, nonlinearly problem and “curse of dimensionality”,
for which SVM has always been the concern of many research scholars. As for application, it has been widely
used in all areas in daily life, such as economic field,
transportation field, medical field and so on. Except for
these traditional areas, SVM also is used for new hot
areas, like mobile multimedia.

"""


result=chain.invoke({'text':text})
print(result)


# visualize the chain 
chain.get_graph().print_ascii()