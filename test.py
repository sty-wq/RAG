import docx

d = docx.Document("sample_notes/柠檬微趣CPP.docx")
for p in d.paragraphs:
    print(repr(p.style.name), "|", p.text)
