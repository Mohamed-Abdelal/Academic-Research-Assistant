# generates research reports in various formats

from typing import List, Optional

from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate


class ReportGenerator:

    def __init__(self, llm):
        self.llm = llm
        self.templates = {
            "summary": PromptTemplate(
                template=(
                    "Summarize the following research information on '{topic}' concisely. "
                    "Include key findings, implications, and source references.\n\n{content}"
                ),
                input_variables=["topic", "content"],
            ),
            "detailed": PromptTemplate(
                template=(
                    "Create a detailed research report on '{topic}' from the following information. "
                    "Include sections for: Introduction, Background, Key Findings, "
                    "Methodology Overview, Detailed Analysis, Practical Implications, "
                    "and Conclusion.\n\n{content}"
                ),
                input_variables=["topic", "content"],
            ),
            "comparative": PromptTemplate(
                template=(
                    "Create a comparative analysis report on '{topic}'. "
                    "Compare and contrast the following research information, "
                    "highlighting areas of agreement, disagreement, and research gaps.\n\n{content}"
                ),
                input_variables=["topic", "content"],
            ),
            "literature_review": PromptTemplate(
                template=(
                    "Write a literature review on '{topic}'. Include: Introduction, "
                    "Thematic Organisation, Historical Context, Current State-of-the-Art, "
                    "Synthesis of Findings, Research Gaps, Future Directions.\n\n{content}"
                ),
                input_variables=["topic", "content"],
            ),
            "executive_summary": PromptTemplate(
                template=(
                    "Write a one-page executive summary on '{topic}'. Include: "
                    "Top 7 Key Findings, Practical Implications, Recommended Actions.\n\n{content}"
                ),
                input_variables=["topic", "content"],
            ),
        }

    def generate_report(self, content: str, topic: str, report_type: str = "summary") -> str:
        template = self.templates.get(report_type)
        if template is None:
            raise ValueError(f"Unknown report type: {report_type}. Available: {list(self.templates.keys())}")
        chain = LLMChain(llm=self.llm, prompt=template)
        return chain.run(topic=topic, content=content)

    def generate_structured_report(
        self, content: str, topic: str, sections: Optional[List[str]] = None
    ) -> str:
        if sections is None:
            sections = ["Introduction", "Methodology", "Findings", "Conclusion"]

        section_results = {}
        for section in sections:
            prompt = PromptTemplate(
                template=(
                    f"Based on the following research information, generate the "
                    f"'{section}' section of a research report on '{{topic}}':\n\n{{content}}"
                ),
                input_variables=["topic", "content"],
            )
            chain = LLMChain(llm=self.llm, prompt=prompt)
            section_results[section] = chain.run(topic=topic, content=content)

        full_report = f"# Research Report: {topic}\n\n"
        for section in sections:
            full_report += f"## {section}\n\n{section_results[section]}\n\n"
        return full_report
