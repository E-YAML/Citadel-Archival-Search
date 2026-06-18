from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.core.config import settings


# --- Pydantic Models for Structured Outputs ---

class GradeDocuments(BaseModel):
    """
    Binary relevance check response.
    """
    is_relevant: bool = Field(
        description="True if the document contains semantic relevance or keywords answering the question, False otherwise."
    )


class GradeHallucinations(BaseModel):
    """
    Binary grounding verification response.
    """
    has_hallucination: bool = Field(
        description="True if the generated answer contains assertions or facts not supported by context documents, False if fully grounded."
    )


class GradeAnswer(BaseModel):
    """
    Binary resolution verification response.
    """
    is_valid: bool = Field(
        description="True if the generated answer directly addresses and resolves the user's question, False otherwise."
    )


# --- LLM Configurations ---

# Standard model for quick evaluation tasks
llm_grade = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model="llama3-8b-8192",
    temperature=0.0
)

# Advanced model for response generation tasks
llm_generate = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model="llama3-70b-8192",
    temperature=0.0
)


# --- Chain Formulations ---

# 1. Document Relevance Grader Chain
retrieval_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an objective grader assessing relevance of a retrieved document to a user question.\n"
               "Analyze the document text. If it contains information, keywords, or semantic meaning relevant to the user question, grade it as relevant.\n"
               "Provide a strict binary decision: True (relevant) or False (irrelevant)."),
    ("human", "Retrieved Document:\n\n{document}\n\nUser Question: {question}")
])
retrieval_grader_chain = retrieval_grader_prompt | llm_grade.with_structured_output(GradeDocuments)


# 2. Hallucination Grader Chain
hallucination_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an objective grader assessing whether an LLM generation is grounded in / supported by a set of retrieved documents.\n"
               "If the generated response contains ANY facts, details, or claims that cannot be directly verified or inferred from the context documents, mark it as containing hallucinations (True).\n"
               "If all claims in the generation are fully supported, mark it as grounded without hallucinations (False)."),
    ("human", "Retrieved Documents Context:\n\n{documents}\n\nLLM Generation:\n\n{generation}")
])
hallucination_grader_chain = hallucination_grader_prompt | llm_grade.with_structured_output(GradeHallucinations)


# 3. Answer Grader Chain
answer_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an objective grader assessing whether a generated answer addresses or resolves the user question.\n"
               "Grade as True if the answer addresses the core question directly and provides a resolution. Grade as False if it is irrelevant, evasive, or fails to address the question."),
    ("human", "User Question: {question}\n\nLLM Generation:\n\n{generation}")
])
answer_grader_chain = answer_grader_prompt | llm_grade.with_structured_output(GradeAnswer)


# 4. Question Query Rewriter Chain
rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert search query optimizer. You are given a user question that failed to return relevant search results.\n"
               "Rewrite the question into a highly optimized search query designed for vector and keyword-based hybrid search. "
               "Focus on core entities, locations, and semantic terms. Return ONLY the rewritten query text, with no introductory or explanatory remarks."),
    ("human", "Failing Question: {question}\nOptimized Query:")
])
question_rewriter_chain = rewriter_prompt | llm_grade | StrOutputParser()


# 5. Citadel Maester Generator Chain
generator_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Maester of the Citadel. Answer the user question ONLY using the provided excerpts from the 'A Song of Ice and Fire' books.\n"
               "You must adhere strictly to these rules:\n"
               "1. For every fact, claim, or statement you make, you MUST explicitly cite the book title and chapter title (from the document metadata).\n"
               "2. If the answer cannot be found in the provided excerpts, say: 'Based on the archival scrolls of the Citadel, I do not possess this knowledge.'\n"
               "3. Do not make up information or introduce external lore outside of the context.\n"
               "4. Format your final response in clean Markdown with citations clearly displayed."),
    ("human", "Context Documents:\n\n{context}\n\nUser Question: {question}")
])
generator_chain = generator_prompt | llm_generate | StrOutputParser()
