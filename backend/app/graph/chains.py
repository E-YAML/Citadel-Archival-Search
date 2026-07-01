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
    model="llama-3.1-8b-instant",
    temperature=0.0
)

# Advanced model for response generation tasks
llm_generate = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.0
)


# --- Chain Formulations ---

# 1. Document Relevance Grader Chain (Text-based, robust)
retrieval_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an objective grader assessing relevance of a retrieved document to a user question.\n"
               "Analyze the document text. If the document contains any details, facts, or mentions related to the characters, entities, or topics in the user question, output ONLY 'yes'. Otherwise, output 'no'.\n"
               "Do not output any other text or explanation. Only output 'yes' or 'no'."),
    ("human", "Retrieved Document:\n\n{document}\n\nUser Question: {question}")
])
retrieval_grader_chain = retrieval_grader_prompt | llm_grade | StrOutputParser()


# 2. Hallucination Grader Chain (Text-based, robust)
hallucination_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an objective grader assessing whether an LLM generation is grounded in / supported by a set of retrieved documents.\n"
               "Analyze the generated response and the context documents. If the generated response is fully grounded in and supported by the context documents, output ONLY 'yes'. If the generated response contains ANY facts, claims, or details that cannot be verified or inferred from the context documents, output ONLY 'no'.\n"
               "Do not output any other text or explanation. Only output 'yes' or 'no'."),
    ("human", "Retrieved Documents Context:\n\n{documents}\n\nLLM Generation:\n\n{generation}")
])
hallucination_grader_chain = hallucination_grader_prompt | llm_generate | StrOutputParser()


# 3. Answer Grader Chain (Text-based, robust)
answer_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an objective grader assessing whether a generated answer addresses or resolves the user question.\n"
               "Analyze the generated response. If the answer addresses the core question directly and provides a resolution, output ONLY 'yes'. Otherwise, output 'no'.\n"
               "Do not output any other text or explanation. Only output 'yes' or 'no'."),
    ("human", "User Question: {question}\n\nLLM Generation:\n\n{generation}")
])
answer_grader_chain = answer_grader_prompt | llm_generate | StrOutputParser()



# 4. Question Query Rewriter Chain
rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert search query optimizer for Westeros archives (A Song of Ice and Fire / Game of Thrones).\n"
               "You are given a user question that failed to return relevant search results.\n"
               "Your goal is to rewrite the question into a highly optimized search query for vector and keyword-based hybrid search of ASOIAF lore.\n"
               "Follow these guidelines:\n"
               "1. Identify and correct any spelling errors, typos, or phonetic/misremembered names or terms (e.g., 'daemon targeryan' -> 'Daemon Targaryen', 'deanerys' -> 'Daenerys', etc.).\n"
               "2. Extract the core entities, characters, and subjects of the query.\n"
               "3. Do NOT add external assumptions, speculative plot details, or hallucinated historical contexts not present in the original question (e.g. do not guess how or when a character died if not asked, as this will overly restrict search results).\n"
               "4. Keep the query clean, simple, and focused on key entities to maximize search coverage.\n"
               "5. Return ONLY the optimized query text itself, with no quotes, introductory, or explanatory remarks."),
    ("human", "Failing Question: {question}\nOptimized Query:")
])
question_rewriter_chain = rewriter_prompt | llm_generate | StrOutputParser()


# 5. Citadel Maester Generator Chain
generator_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Maester of the Citadel. Answer the user question ONLY using the provided Context Documents.\n"
               "CRITICAL: Do NOT introduce any external facts, characters, details, or lore not explicitly mentioned in the provided Context Documents.\n"
               "Your answer must be 100% grounded in the Context Documents. If a detail is not explicitly written in the Context Documents, you must treat it as completely unknown and never mention it.\n"
               "Rules:\n"
               "1. For every fact, claim, or statement you make, you MUST explicitly cite the book title and chapter title (from the document metadata).\n"
               "2. If the answer cannot be found in the provided excerpts, say: 'Based on the archival scrolls of the Citadel, I do not possess this knowledge.'\n"
               "3. Format your final response in clean Markdown with citations clearly displayed."),
    ("human", "Context Documents:\n\n{context}\n\nUser Question: {question}")
])
generator_chain = (generator_prompt | llm_generate | StrOutputParser()).with_config({"tags": ["citadel_generation"]})
