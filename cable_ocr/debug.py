from PIL import ImageDraw


def overlay(image,tables,items,path,mapping=None,coordinate_known=True):
    view=image.copy(); d=ImageDraw.Draw(view)
    for t in tables:
        for row in t['rows']:
            for c in row['cells']:
                b=c['bbox']; d.rectangle(b,outline='#008744',width=2)
                d.text((b[0]+3,b[1]+3),f"r{c['row']}c{c['column']}",fill='#008744')
    if coordinate_known:
        mapped={r['fragment_id']:r for r in (mapping or {}).get('records',[])}
        for item in items:
            b=item.get('bbox')
            if b:
                record=mapped.get(item['id']); color='#1565c0' if not mapping or record and record['cell_ids'] else '#d50000'
                d.rectangle(b,outline=color,width=2)
                label=item['id']
                if record and record['cell_ids']:label+=' -> '+','.join(x.split('-')[-2]+'/'+x.split('-')[-1] for x in record['cell_ids'])
                d.text((b[0],max(0,b[1]-13)),label,fill=color)
    else:
        d.rectangle((0,0,image.width,28),fill='white')
        d.text((5,7),'COORDINATE_SPACE_UNKNOWN: provider boxes not overlaid',fill='#d50000')
    view.save(path)
