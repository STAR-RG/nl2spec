from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from nl2spec.core.convert.ere import (
    extract_events,
    extract_ere_expression,
    extract_signature,
    extract_violation_block,
)


class ERENL:
    """
    MOP -> NL renderer for ERE specifications.

    Reads the .mop source directly and fills the ERE task template
    without converting the specification to JSON IR first.
    """

    def extract_context(
        self,
        source: Union[str, Path],
        domain: Optional[str] = None,
        spec_id: Optional[str] = None,
    ) -> dict:
        mop_text, resolved_spec_id, resolved_domain = self._read_mop_source(
            source=source,
            domain=domain,
            spec_id=spec_id,
        )

        signature = extract_signature(mop_text)
        methods = extract_events(mop_text)
        declared_events = {m["name"] for m in methods}
        ere_expression = extract_ere_expression(
            mop_text,
            declared_events=declared_events,
        )
        violation = extract_violation_block(mop_text)

        return {
            "SPEC_ID": resolved_spec_id,
            "SIGNATURE_PARAMETERS": self._render_signature_parameters(signature),
            "EVENT_BLOCK": self._render_events_block(methods),
            "ERE_BLOCK": self._render_ere_block(ere_expression),
            "VIOLATION_TAG": self._render_violation_tag(violation),
            "VIOLATION_STATEMENTS": self._render_violation_statements(violation),
            "HAS_RESET": self._render_has_reset(violation),
            "DOMAIN": resolved_domain,
        }

    # ==========================================================
    # SOURCE
    # ==========================================================

    def _read_mop_source(
        self,
        source: Union[str, Path],
        domain: Optional[str],
        spec_id: Optional[str],
    ) -> tuple:
        if isinstance(source, Path):
            mop_path = source
        elif isinstance(source, str):
            mop_path = Path(source)
        else:
            raise TypeError("ERENL.extract_context expected str or Path.")

        if not mop_path.exists() or not mop_path.is_file():
            raise FileNotFoundError(f"MOP file not found: {mop_path}")

        mop_text = mop_path.read_text(encoding="utf-8", errors="replace")
        resolved_spec_id = spec_id or mop_path.stem
        resolved_domain = domain or self._detect_domain_from_path(mop_path)

        return mop_text, resolved_spec_id, resolved_domain

    def _detect_domain_from_path(self, path: Path) -> str:
        for part in path.parts:
            if part in {"io", "lang", "util", "net"}:
                return part
        return "unknown"

    # ==========================================================
    # SIGNATURE
    # ==========================================================

    def _render_signature_parameters(self, signature: Dict[str, Any]) -> str:
        params = signature.get("parameters", []) if isinstance(signature, dict) else []

        if not isinstance(params, list) or not params:
            return "none"

        rendered = []
        for p in params:
            if not isinstance(p, dict):
                continue
            ptype = p.get("type", "<?>")
            pname = p.get("name", "<?>")
            rendered.append(f"{ptype} {pname}")

        return ", ".join(rendered) if rendered else "none"

    # ==========================================================
    # EVENTS
    # ==========================================================

    def _render_events_block(self, methods: List[Dict[str, Any]]) -> str:
        if not methods:
            return "No monitored events were provided."

        lines = []

        for idx, ev in enumerate(methods, 1):
            name = ev.get("name", f"event_{idx}")
            action = (ev.get("action") or "event").strip()
            timing = (ev.get("timing") or "").strip()
            params = self._format_parameters(ev.get("parameters"))
            returning = self._format_returning(ev.get("returning"))
            pointcut = self._render_pointcut(ev)
            article = self._article_for(action)

            if returning:
                lines.append(
                    f"{idx}) {name} is observed {timing} the call as {article} {action}. "
                    f"It receives {params}, returns {returning}, and matches:"
                )
            else:
                lines.append(
                    f"{idx}) {name} is observed {timing} the call as {article} {action}. "
                    f"It receives {params} and matches:"
                )

            lines.append(
                "   Top-level operators between consecutive pointcut atoms: "
                f"{self._render_operations_inline(ev)}"
            )
            lines.append(f"   {pointcut}")
            lines.append("")

        return "\n".join(lines).strip()

    def _render_operations_inline(self, ev: Dict[str, Any]) -> str:
        ops = ev.get("operation", []) or []
        return ", ".join(ops) if ops else "none"

    def _render_pointcut(self, ev: Dict[str, Any]) -> str:
        funcs = ev.get("function", []) or []
        ops = ev.get("operation", []) or []

        if not funcs:
            return "<missing pointcut>"

        pieces = []

        for i, fn in enumerate(funcs):
            if not isinstance(fn, dict):
                continue

            fname = fn.get("name", "unknown")
            args = fn.get("arguments", []) or []
            negated = fn.get("negated", False)

            inner_parts = []
            for a in args:
                if not isinstance(a, dict):
                    continue
                value = a.get("value", "")
                if value:
                    inner_parts.append(value)

            inner = ", ".join(inner_parts)
            atom = f"{fname}({inner})" if fname != "unknown" else inner

            if negated:
                atom = f"!{atom}"

            pieces.append(atom)

            if i < len(ops):
                pieces.append(ops[i])

        return " ".join(pieces).strip()

    def _format_parameters(self, params) -> str:
        if not params:
            return "none"

        rendered = []
        for p in params:
            if not isinstance(p, dict):
                continue
            rendered.append(f'{p.get("type", "<?>")} {p.get("name", "<?>")}')

        return ", ".join(rendered) if rendered else "none"

    def _format_returning(self, returning):
        if not returning or not isinstance(returning, dict):
            return None

        rtype = (returning.get("type") or "").strip()
        rname = (returning.get("name") or "").strip()

        if not rtype and not rname:
            return None

        return f"{rtype} {rname}".strip()

    def _article_for(self, text: str) -> str:
        if not text:
            return "a"
        return "an" if text[0].lower() in {"a", "e", "i", "o", "u"} else "a"

    # ==========================================================
    # ERE
    # ==========================================================

    def _render_ere_block(self, ere_expression: str) -> str:
        expression = (ere_expression or "").strip()

        if not expression:
            return "No ERE expression was provided."

        return expression

    # ==========================================================
    # VIOLATION
    # ==========================================================

    def _render_violation_tag(self, violation: Dict[str, Any]) -> str:
        return (violation.get("tag") or "").strip()

    def _render_violation_statements(self, violation: Dict[str, Any]) -> str:
        body = violation.get("body", {}) or {}
        statements = body.get("statements", []) or []

        if not statements:
            return "none"

        lines = []

        for idx, stmt in enumerate(statements, 1):
            if not isinstance(stmt, dict):
                continue

            stype = stmt.get("type", "")

            if stype == "log":
                level = stmt.get("level", "")
                message = stmt.get("message", "")
                lines.append(f"{idx}) type=log; level={level}; message={message}")

            elif stype == "command":
                name = stmt.get("name", "")
                lines.append(f"{idx}) type=command; name={name}")

            elif stype == "raw":
                value = stmt.get("value", "")
                lines.append(f"{idx}) type=raw; value={value}")

            else:
                lines.append(f"{idx}) type={stype}")

        return "\n".join(lines) if lines else "none"

    def _render_has_reset(self, violation: Dict[str, Any]) -> str:
        body = violation.get("body", {}) or {}
        return str(bool(body.get("has_reset", False))).lower()