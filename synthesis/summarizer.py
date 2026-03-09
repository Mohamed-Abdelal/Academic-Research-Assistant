# synthesizes docs with LangChain chains

from typing import List

from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


class InformationSynthesizer:

    def __init__(self, llm):
        self.llm = llm

    def create_summary_chain(self, chain_type: str = "map_reduce"):
        return load_summarize_chain(
            llm=self.llm,
            chain_type=chain_type,
            verbose=False,
        )

    def synthesize_documents(self, documents: List[Document], chain_type: str = "stuff") -> str:
        if not documents:
            return "No documents to synthesize."
        chain = self.create_summary_chain(chain_type)
        return chain.invoke(documents)["output_text"]

    def synthesize_with_query_focus(self, documents: List[Document], query: str) -> str:
        if not documents:
            return "No documents to synthesize."

        query_prompt = PromptTemplate(
            template=(
                "Synthesize the following research information to answer this query: {query}\n\n"
                "Information:\n{text}\n\n"
                "Provide a thorough, well-structured synthesis with inline citations [Author, Year]."
            ),
            input_variables=["query", "text"],
        )
        query_chain = LLMChain(llm=self.llm, prompt=query_prompt)

        partial_results = []
        for doc in documents[:6]:
            meta = doc.metadata
            source_label = f"[{meta.get('authors', 'Unknown')}, {meta.get('year', 'n.d.')}]"
            result = query_chain.run(query=query, text=f"{source_label}: {doc.page_content}")
            partial_results.append(result)

        combine_prompt = PromptTemplate(
            template=(
                "Combine these partial research syntheses into one comprehensive, "
                "well-structured response to: {query}\n\n"
                "Partial syntheses:\n{answers}\n\n"
                "Requirements:\n"
                "- Use [Author, Year] citations\n"
                "- Highlight consensus and disagreements\n"
                "- Include practical implications\n"
                "- Suggest research gaps"
            ),
            input_variables=["query", "answers"],
        )
        combine_chain = LLMChain(llm=self.llm, prompt=combine_prompt)
        return combine_chain.run(query=query, answers="\n\n---\n\n".join(partial_results))
