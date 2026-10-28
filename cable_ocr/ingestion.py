"""Canonical pixels and native extraction; independent of OCR availability."""
from pathlib import Path
import unicodedata
from PIL import Image, ImageOps
from .common import PhaseError, digest, save_json
from .admission import MAX_IMAGE_PIXELS


def ingest(path, page_index, out, dpi=144):
    path, out = Path(path), Path(out)
    if not path.is_file():
        raise PhaseError("INPUT_NOT_FOUND")
    if path.stat().st_size > 100 * 1024 * 1024:
        raise PhaseError("INPUT_TOO_LARGE")
    if page_index < 0 or not 72 <= dpi <= 300:
        raise PhaseError("INVALID_PAGE_OR_DPI")
    data = path.read_bytes()
    out.mkdir(parents=True, exist_ok=True)
    native, observation = [], {}
    meta = {"page_index": page_index, "input_hash": digest(data), "rotation": 0,
            "coordinate_space": "canonical_page_pixels", "origin": "top_left", "x": "right", "y": "down"}
    if data.startswith(b"%PDF-"):
        import pdfplumber
        import pypdfium2 as pdfium
        try:
            doc = pdfium.PdfDocument(data)
            if page_index >= len(doc):
                raise PhaseError("PAGE_OUT_OF_RANGE")
            page = doc[page_index]
            pdf_size = list(page.get_size())
            if pdf_size[0]*pdf_size[1]*(dpi/72)**2 > MAX_IMAGE_PIXELS:
                raise PhaseError("IMAGE_DIMENSION_EXCEEDED")
            bitmap = page.render(scale=dpi/72)
            image = bitmap.to_pil().convert("RGB")
            bitmap.close(); page.close(); doc.close()
            with pdfplumber.open(path) as pdf:
                p = pdf.pages[page_index]
                sx, sy = image.width/p.width, image.height/p.height
                meta.update(kind="pdf", pdf_size_points=pdf_size, pdfplumber_size=[p.width,p.height],
                            rotation=p.rotation, pdf_bbox=list(p.bbox), page_count=len(pdf.pages),
                            render_config={"backend":"pypdfium2", "dpi":dpi},
                            native_to_canonical=[sx,sy,-p.bbox[0]*sx,-p.bbox[1]*sy])
                chars = p.chars
                useful = [c for c in chars if c.get("text", "").strip()]
                bad = sum(any(unicodedata.category(x).startswith("C") or x=="\ufffd" for x in c.get("text", "")) for c in useful)
                image_area = sum(max(0,x["x1"]-x["x0"])*max(0,x["bottom"]-x["top"]) for x in p.images)
                image_ratio = min(1.0, image_area/(p.width*p.height))
                coverage = sum(max(0,c["x1"]-c["x0"])*max(0,c["bottom"]-c["top"]) for c in useful)/(p.width*p.height)
                # Conservative PoC: full-page scanned image + hidden text is not trusted.
                native_geometry_ok=abs(pdf_size[0]-p.width)<.1 and abs(pdf_size[1]-p.height)<.1
                reliable = len(useful) >= 10 and bad == 0 and image_ratio < .5 and p.rotation == 0 and native_geometry_ok
                observation = {"text_object_count":len(chars),"nonblank_char_count":len(useful),
                               "invalid_char_count":bad,"text_box_area_ratio":coverage,"image_area_ratio":image_ratio,
                               "native_render_geometry_consistent":native_geometry_ok}
                for i,c in enumerate(chars):
                    b=[(c['x0']-p.bbox[0])*sx,(c['top']-p.bbox[1])*sy,
                       (c['x1']-p.bbox[0])*sx,(c['bottom']-p.bbox[1])*sy]
                    native.append({"id":f"native-{i}","text":c['text'],"bbox":b,"confidence":None,"characters":[]})
                meta["source_decision"] = "NATIVE_TEXT_RELIABLE" if reliable else "OCR_REQUIRED"
                meta["source_reasons"] = ["conservative_native_character_and_image_gate" if reliable else "native_evidence_not_confirmed"]
        except PhaseError:
            raise
        except Exception as e:
            raise PhaseError("PDF_INGESTION_FAILED", {"exception_type":type(e).__name__}) from None
    else:
        if page_index != 0:
            raise PhaseError("PAGE_OUT_OF_RANGE")
        try:
            with Image.open(path) as source:
                if source.format not in {"JPEG","PNG"}:
                    raise PhaseError("UNSUPPORTED_INPUT")
                if source.width*source.height > MAX_IMAGE_PIXELS:
                    raise PhaseError("IMAGE_DIMENSION_EXCEEDED")
                orientation = source.getexif().get(274,1)
                image = ImageOps.exif_transpose(source).convert("RGB")
                meta.update(kind="image",page_count=1, exif_orientation=orientation,
                            source_decision="OCR_REQUIRED", source_reasons=["image_input"],
                            render_config={"backend":"Pillow", "exif_transpose":True})
        except PhaseError:
            raise
        except Exception as e:
            raise PhaseError("IMAGE_INGESTION_FAILED", {"exception_type":type(e).__name__}) from None
    image.save(out/"canonical_page.png")
    meta.update(width=image.width,height=image.height,image_hash=digest((out/"canonical_page.png").read_bytes()),
                canonical_image_ref="canonical_page.png", observation=observation)
    save_json(out/"page_observation.json", meta)
    save_json(out/"native_evidence.json", {"items":native,"source_hash":meta["input_hash"]})
    return image,meta,native
