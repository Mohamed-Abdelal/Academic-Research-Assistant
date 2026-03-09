# mock paper database across 10 domains

from typing import List, Dict
from langchain_core.documents import Document

MOCK_PAPERS: Dict[str, List[Dict]] = {
    "machine learning": [
        {"title": "Attention Is All You Need", "authors": "Vaswani et al.", "year": 2017, "journal": "NeurIPS", "abstract": "We propose the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions. Experiments on two machine translation tasks show these models are superior in quality while being more parallelizable and requiring significantly less time to train.", "citations": 90_000, "doi": "10.48550/arXiv.1706.03762", "relevance": 0.97},
        {"title": "BERT: Pre-training of Deep Bidirectional Transformers", "authors": "Devlin et al.", "year": 2018, "journal": "NAACL", "abstract": "We introduce BERT, designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.", "citations": 75_000, "doi": "10.48550/arXiv.1810.04805", "relevance": 0.91},
        {"title": "Deep Residual Learning for Image Recognition", "authors": "He et al.", "year": 2016, "journal": "CVPR", "abstract": "We present a residual learning framework to ease the training of networks substantially deeper than those previously used.", "citations": 120_000, "doi": "10.1109/CVPR.2016.90", "relevance": 0.85},
        {"title": "Scaling Laws for Neural Language Models", "authors": "Kaplan et al. (OpenAI)", "year": 2020, "journal": "arXiv", "abstract": "We study empirical scaling laws for language model performance on the cross-entropy loss.", "citations": 8_500, "doi": "10.48550/arXiv.2001.08361", "relevance": 0.88},
    ],
    "artificial intelligence": [
        {"title": "GPT-4 Technical Report", "authors": "OpenAI", "year": 2023, "journal": "arXiv", "abstract": "We report the development of GPT-4, a large-scale multimodal model which can accept image and text inputs and produce text outputs.", "citations": 11_000, "doi": "10.48550/arXiv.2303.08774", "relevance": 0.95},
        {"title": "Human-Level Control through Deep Reinforcement Learning", "authors": "Mnih et al. — DeepMind", "year": 2015, "journal": "Nature", "abstract": "We present a model-free reinforcement learning algorithm that can learn to play Atari 2600 video games directly from pixels.", "citations": 18_000, "doi": "10.1038/nature14236", "relevance": 0.89},
        {"title": "LLM-based Autonomous Agents: A Survey", "authors": "Wang et al.", "year": 2024, "journal": "arXiv", "abstract": "We survey LLM-based autonomous agents that leverage large language models as the central controller.", "citations": 3_200, "doi": "10.48550/arXiv.2308.11432", "relevance": 0.92},
    ],
    "climate change": [
        {"title": "Global Warming of 1.5C — IPCC Special Report", "authors": "IPCC Working Group I", "year": 2018, "journal": "IPCC", "abstract": "An IPCC special report on the impacts of global warming of 1.5C above pre-industrial levels.", "citations": 28_000, "doi": "10.1017/9781009157940", "relevance": 0.98},
        {"title": "Carbon Capture and Storage: A Key Technology", "authors": "Bui et al.", "year": 2022, "journal": "Nature Energy", "abstract": "Carbon capture and storage technologies are critical for achieving net-zero emissions.", "citations": 4_100, "doi": "10.1038/s41560-022-01140-6", "relevance": 0.87},
        {"title": "Renewable Energy Transition: Evidence from 150 Countries", "authors": "Gielen et al.", "year": 2021, "journal": "Science", "abstract": "We model the global energy transition and find that renewables can supply 86% of global energy by 2050.", "citations": 6_800, "doi": "10.1016/j.esr.2019.100290", "relevance": 0.90},
    ],
    "quantum computing": [
        {"title": "Quantum Supremacy Using a Programmable Superconducting Processor", "authors": "Arute et al. — Google AI", "year": 2019, "journal": "Nature", "abstract": "We developed a quantum processor using 53 programmable superconducting qubits.", "citations": 6_800, "doi": "10.1038/s41586-019-1666-5", "relevance": 0.97},
        {"title": "Quantum Error Correction Below Threshold", "authors": "Google Quantum AI", "year": 2023, "journal": "Nature", "abstract": "We demonstrate a quantum memory with below-threshold logical error rates.", "citations": 1_200, "doi": "10.1038/s41586-023-06096-3", "relevance": 0.93},
    ],
    "blockchain": [
        {"title": "Bitcoin: A Peer-to-Peer Electronic Cash System", "authors": "Nakamoto, S.", "year": 2008, "journal": "White Paper", "abstract": "A purely peer-to-peer version of electronic cash.", "citations": 22_000, "doi": "https://bitcoin.org/bitcoin.pdf", "relevance": 0.99},
        {"title": "Ethereum: A Next-Generation Smart Contract Platform", "authors": "Buterin, V.", "year": 2014, "journal": "White Paper", "abstract": "We propose a blockchain with a built-in Turing-complete programming language.", "citations": 12_000, "doi": "https://ethereum.org/whitepaper", "relevance": 0.94},
    ],
    "healthcare": [
        {"title": "Deep Learning for Medical Image Analysis", "authors": "Litjens et al.", "year": 2022, "journal": "Medical Image Analysis", "abstract": "We review deep learning applications in medical image analysis.", "citations": 9_500, "doi": "10.1016/j.media.2017.07.005", "relevance": 0.93},
        {"title": "Large Language Models in Clinical Medicine", "authors": "Singhal et al. — Google", "year": 2023, "journal": "Nature Medicine", "abstract": "We introduce Med-PaLM, a large language model designed for safe medical question answering.", "citations": 3_400, "doi": "10.1038/s41591-023-02476-5", "relevance": 0.95},
    ],
    "cybersecurity": [
        {"title": "A Survey of Phishing Attacks and Countermeasures", "authors": "Alabdan, R.", "year": 2022, "journal": "IEEE Access", "abstract": "Comprehensive review of phishing attacks and detection methods.", "citations": 2_100, "doi": "10.1109/ACCESS.2020.2983829", "relevance": 0.89},
        {"title": "Zero-Trust Architecture: NIST SP 800-207", "authors": "Rose et al. — NIST", "year": 2020, "journal": "NIST", "abstract": "This document defines zero trust architecture and its key principles.", "citations": 5_700, "doi": "10.6028/NIST.SP.800-207", "relevance": 0.92},
    ],
    "education": [
        {"title": "The Role of AI in Personalised Learning", "authors": "VanLehn, K.", "year": 2023, "journal": "Educational Psychology Review", "abstract": "AI-powered intelligent tutoring systems achieve learning gains of 2 sigma.", "citations": 4_300, "doi": "10.1007/s10648-023-09788-z", "relevance": 0.91},
        {"title": "ChatGPT in Higher Education: Opportunities and Risks", "authors": "Tlili et al.", "year": 2023, "journal": "Computers & Education", "abstract": "We survey 2000 students and 500 faculty on perceptions of ChatGPT.", "citations": 2_800, "doi": "10.1016/j.compedu.2023.104811", "relevance": 0.88},
    ],
    "robotics": [
        {"title": "RT-2: Vision-Language-Action Models", "authors": "Brohan et al. — Google DeepMind", "year": 2023, "journal": "arXiv", "abstract": "We propose RT-2, transferring knowledge from vision-language models to robot control.", "citations": 1_900, "doi": "10.48550/arXiv.2307.15818", "relevance": 0.94},
    ],
    "neuroscience": [
        {"title": "The Human Connectome Project", "authors": "Van Essen et al.", "year": 2021, "journal": "NeuroImage", "abstract": "The HCP aims to map complete structural and functional neural connections in vivo.", "citations": 7_200, "doi": "10.1016/j.neuroimage.2012.02.018", "relevance": 0.90},
    ],
}

DEVELOPMENTS = {
    "machine learning": "2024 highlights: (1) Mixture-of-Experts reducing inference costs 10x. (2) Constitutional AI improving alignment. (3) LoRA enabling efficient fine-tuning. (4) Mamba architecture challenging Transformers.",
    "artificial intelligence": "2024 highlights: (1) GPT-4o multimodal capabilities. (2) AI agents completing complex tasks. (3) RAG becoming industry standard. (4) Open-source LLMs matching proprietary models.",
    "climate change": "2024 highlights: (1) 2023 hottest year on record. (2) Solar now cheapest electricity source. (3) Carbon capture costs dropped 40%. (4) Arctic sea ice at historic low.",
    "quantum computing": "2024 highlights: (1) IBM Condor 1000+ qubit processor. (2) Google below-threshold error correction. (3) Microsoft topological qubit approach. (4) Post-quantum NIST standards finalised.",
    "blockchain": "2024 highlights: (1) Ethereum PoS reduces energy 99.9%. (2) Layer-2 networks handle 10x TPS. (3) ZK-proofs enabling private DeFi. (4) Bitcoin ETF approval.",
    "healthcare": "2024 highlights: (1) AI outperforms radiologists in cancer detection. (2) Med-PaLM 2 passes USMLE. (3) mRNA vaccine platforms expanded. (4) Wearable biosensors for continuous monitoring.",
    "cybersecurity": "2024 highlights: (1) AI-powered phishing surged 200%. (2) Zero-trust adoption accelerated. (3) Post-quantum cryptography migration begins. (4) EU AI Act security requirements.",
    "education": "2024 highlights: (1) 85% of universities have AI policies. (2) AI tutors showing 2 sigma improvement. (3) Micro-credentials growing 40% annually. (4) Adaptive learning at scale.",
    "robotics": "2024 highlights: (1) Humanoid robots entering factories. (2) Vision-language-action models. (3) Boston Dynamics Atlas performing tasks. (4) Medical robots sub-millimetre precision.",
    "neuroscience": "2024 highlights: (1) BCIs restore communication for ALS. (2) Human Connectome Phase II complete. (3) AI predicting neural activity 95%. (4) Optogenetics treating depression.",
}


class PaperDatabaseTool:

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        q = query.lower()
        results = []
        for topic, papers in MOCK_PAPERS.items():
            if any(w in q for w in topic.split()):
                results.extend(papers)
        if not results:
            results = MOCK_PAPERS["artificial intelligence"] + MOCK_PAPERS["machine learning"][:1]
        results.sort(key=lambda x: x.get("relevance", 0.5), reverse=True)
        return results[:num_results]

    def search_to_documents(self, query: str, num_results: int = 5) -> List[Document]:
        papers = self.search(query, num_results)
        return [
            Document(
                page_content=p.get("abstract", ""),
                metadata={
                    "title": p["title"],
                    "authors": p["authors"],
                    "year": p["year"],
                    "journal": p["journal"],
                    "citations": p.get("citations", 0),
                    "doi": p.get("doi", ""),
                    "source_type": "academic",
                    "relevance": p.get("relevance", 0.5),
                },
            )
            for p in papers
        ]

    def get_developments(self, query: str) -> str:
        q = query.lower()
        for key, dev in DEVELOPMENTS.items():
            if key in q or any(w in q for w in key.split()):
                return dev
        return "Active research area with substantial recent publications."
