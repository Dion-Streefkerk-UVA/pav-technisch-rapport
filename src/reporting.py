# tabellen wegschrijven en findings.md opbouwen.
# elke tabel gaat naar csv (om te checken) en .tex (voor het latex-verslag).
from . import config


def save_table(df, name, caption, label, index=False, float_format="%.3f"):
    # tabel opslaan als csv én .tex, geeft beide paden terug
    csv_path = config.TABLES_DIR / f"{name}.csv"
    tex_path = config.TABLES_DIR / f"{name}.tex"

    df.to_csv(csv_path, index=index)

    latex = df.to_latex(
        index=index,
        caption=caption,
        label=label,
        float_format=float_format,
        escape=True,
        position="htbp",
    )
    tex_path.write_text(latex, encoding="utf-8")
    return csv_path, tex_path


class FindingsWriter:
    # verzamelt bevindingen per sectie en schrijft ze aan het eind naar findings.md
    def __init__(self):
        # _sections = teksten per sectie, _order = volgorde van de secties
        self._sections = {}
        self._order = []

    def add(self, section, text):
        # tekst aan een sectie toevoegen; nieuwe sectie -> aanmaken en onthouden
        if section not in self._sections:
            self._sections[section] = []
            self._order.append(section)
        self._sections[section].append(text)

    def write(self, path=None):
        # alles naar één markdown-bestand; geen pad -> standaardlocatie uit config
        path = path or config.FINDINGS_FILE
        lines = [
            "# Findings",
            "",
            "_Automatisch gegenereerd door `python main.py` — alle getallen komen "
            "rechtstreeks uit de daadwerkelijke pipeline-run._",
            "",
        ]
        for section in self._order:
            lines.append(f"## {section}")
            lines.append("")
            lines.extend(self._sections[section])
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
