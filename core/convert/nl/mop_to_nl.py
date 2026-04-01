from pathlib import Path
from typing import Optional
import shutil

from nl2spec.core.convert.mop_to_ir import detect_domain, detect_formalism
from nl2spec.core.convert.nl.ere_nl_mop import ERENL
from nl2spec.logging_utils import get_logger


log = get_logger(__name__)


class MOPToNL:
    """
    MOP -> NL converter.

    Current version supports only ERE specifications.
    Other formalisms are skipped during batch generation.
    """

    def __init__(self, template_dir: Path):
        self.template_dir = Path(template_dir)
        self.ere_nl = ERENL()

    # ==========================================================
    # BATCH GENERATION
    # ==========================================================

    def generate_from_directory(
        self,
        mop_root: Path,
        nl_root: Path,
    ) -> int:
        mop_root = Path(mop_root)
        nl_root = Path(nl_root)

        if not mop_root.exists():
            raise FileNotFoundError(f"MOP directory not found: {mop_root}")

        if nl_root.exists():
            shutil.rmtree(nl_root)

        nl_root.mkdir(parents=True, exist_ok=True)

        total = 0
        skipped = 0

        for mop_file in mop_root.rglob("*.mop"):
            text = mop_file.read_text(encoding="utf-8", errors="replace")
            formalism = detect_formalism(text)

            if formalism != "ere":
                skipped += 1
                log.info("[SKIP] file=%s | formalism=%s", mop_file.name, formalism)
                continue

            domain = (detect_domain(mop_file) or "other").lower()
            out_dir = nl_root / domain
            out_dir.mkdir(parents=True, exist_ok=True)

            task = self.build_task(
                mop_file=mop_file,
                domain=domain,
                formalism=formalism,
            )

            out_file = out_dir / f"{mop_file.stem}.txt"
            out_file.write_text(task, encoding="utf-8")

            total += 1
            log.info("[OK] generated=%s", out_file)

        log.info(
            "MOP -> NL finished | generated=%d | skipped_non_ere=%d",
            total,
            skipped,
        )

        return total

    # ==========================================================
    # SINGLE FILE GENERATION
    # ==========================================================

    def build_task(
        self,
        mop_file: Path,
        domain: Optional[str] = None,
        formalism: Optional[str] = None,
    ) -> str:
        mop_file = Path(mop_file)

        if not mop_file.exists():
            raise FileNotFoundError(f"MOP file not found: {mop_file}")

        text = mop_file.read_text(encoding="utf-8", errors="replace")
        detected_formalism = (formalism or detect_formalism(text) or "").lower()

        if detected_formalism != "ere":
            raise ValueError(
                f"This MOPToNL version currently supports only ERE, got: {detected_formalism}"
            )

        resolved_domain = (domain or detect_domain(mop_file) or "other").lower()

        context = self.ere_nl.extract_context(
            source=mop_file,
            domain=resolved_domain,
        )

        return self._render_template("ere", context)

    # ==========================================================
    # TEMPLATE RENDERING
    # ==========================================================

    def _render_template(self, formalism: str, context: dict) -> str:
        template_path = self.template_dir / formalism / f"task_{formalism}.txt"

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        template = template_path.read_text(encoding="utf-8")

        for key, value in context.items():
            template = template.replace(f"{{{{{key}}}}}", str(value or ""))

        return template.strip() + "\n"