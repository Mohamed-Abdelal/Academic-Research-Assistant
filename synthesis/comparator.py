# compares findings across sources

from typing import List

from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


class InformationComparator:

    def __init__(self, llm):
        self.llm = llm

    def compare_sources(self, documents: List[Document], topic: str) -> str:
        if not documents:
            return "No documents to compare."

        sources_text = ""
        for i, doc in enumerate(documents[:6], 1):
            meta = doc.metadata
            sources_text += (
                f"\n[Source {i}] {meta.get('title', 'Unknown')} "
                f"— {meta.get('authors', 'Unknown')} ({meta.get('year', 'n.d.')})\n"
                f"{doc.page_content[:400]}\n"
            )

        compare_prompt = PromptTemplate(
            template=(
                "Compare and contrast the following research sources on: {topic}\n\n"
                "{sources}\n\n"
                "Provide:\n"
                "1. Areas of consensus\n"
                "2. Key disagreements or different perspectives\n"
                "3. Methodological differences\n"
                "4. A synthesis of the overall state of research\n"
                "Use [Source N] citations throughout."
            ),
            input_variables=["topic", "sources"],
        )
        chain = LLMChain(llm=self.llm, prompt=compare_prompt)
        return chain.run(topic=topic, sources=sources_text)
