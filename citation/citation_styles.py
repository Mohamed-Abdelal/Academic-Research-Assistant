# APA, MLA, Chicago formatters

from typing import Dict


def format_apa(source: Dict) -> str:
    authors = source.get("authors", "Unknown")
    year = source.get("year", "n.d.")
    title = source.get("title", "Untitled")
    journal = source.get("journal", "")
    doi = source.get("doi", "")
    cite = f"{authors} ({year}). {title}. *{journal}*."
    if doi:
        cite += f" https://doi.org/{doi}" if not doi.startswith("http") else f" {doi}"
    return cite


def format_mla(source: Dict) -> str:
    authors = source.get("authors", "Unknown")
    title = source.get("title", "Untitled")
    journal = source.get("journal", "")
    year = source.get("year", "n.d.")
    doi = source.get("doi", "")
    cite = f'{authors}. "{title}." *{journal}*, {year}.'
    if doi:
        link = f"https://doi.org/{doi}" if not doi.startswith("http") else doi
        cite += f" {link}."
    return cite


def format_chicago(source: Dict) -> str:
    authors = source.get("authors", "Unknown")
    title = source.get("title", "Untitled")
    journal = source.get("journal", "")
    year = source.get("year", "n.d.")
    doi = source.get("doi", "")
    cite = f'{authors}. "{title}." *{journal}* ({year}).'
    if doi:
        link = f"https://doi.org/{doi}" if not doi.startswith("http") else doi
        cite += f" {link}."
    return cite


STYLES = {
    "apa": format_apa,
    "mla": format_mla,
    "chicago": format_chicago,
}


def list_styles() -> str:
    return "Available citation styles: APA, MLA, Chicago"
