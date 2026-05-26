#let signField() = {
    set text(size: 8pt, fill: gray)
    grid(
        columns: (1fr, auto, 60%),
        column-gutter: 2pt,

        [#v(1em)#line(length: 100%, stroke: 0.3pt)],
        text(size: 12pt)[/],
        [#v(1em)#line(length: 100%, stroke: 0.3pt)],

        [(подпись)], [], [(Ф.И.О.)],
    )
}

#set align(center)
#set par(spacing: 0.5em, first-line-indent: 0cm, justify: false)

*КОМИТЕТ ПО НАУКЕ И ВЫСШЕЙ ШКОЛЫ*

#v(1em)
*САНКТ–ПЕТЕРБУРГСКОЕ ГОСУДАРСТВЕННОЕ БЮДЖЕТНОЕ ПРОФЕССИОНАЛЬНОЕ ОБРАЗОВАТЕЛЬНОЕ
УЧРЕЖДЕНИЕ «АКАДЕМИЯ ИНЖЕНЕРНЫХ ТЕХНОЛОГИЙ И УПРАВЛЕНИЯ»*

#v(3em)
*ПОЯСНИТЕЛЬНАЯ ЗАПИСКА*

*к дипломному проекту*

#v(1em)

#align(left)[
    #set text(style: "italic")
    #grid(
        columns: (auto, 1fr),
        rows: (auto, auto, auto),
        column-gutter: 1em,
        row-gutter: 0.5em,
        grid.cell(rowspan: 3)[Тема:],
        grid.cell(stroke: (bottom: 0.5pt + black))[
            Разработка программы детекции текста для локализации
        ],
        grid.cell(stroke: (bottom: 0.5pt + black))[
            японских комиксов (манги) с сохранением исходной графики на
        ],
        grid.cell(stroke: (bottom: 0.5pt + black))[
            основе нейросетевого перевода
        ],
    )
]

#[
    #set text(style: "italic")
    #grid(
        columns: (1fr, 1fr, 1fr),
        rows: auto,
        column-gutter: 1em,
        row-gutter: 0.5em,
        text(style: "normal")[*Руководитель*],
        [],
        [],
        grid.cell(stroke: (bottom: 0.5pt + black))[-],
        grid.cell(stroke: (bottom: 0.5pt + black))[],
        grid.cell(stroke: (bottom: 0.5pt + black))[-],
        [(должность)],
        [(подпись)],
        [(И. О. Фамилия)],

        text(style: "normal")[*Студент*],
        [],
        [],
        grid.cell(stroke: (bottom: 0.5pt + black))[9ИС-31],
        grid.cell(stroke: (bottom: 0.5pt + black))[],
        grid.cell(stroke: (bottom: 0.5pt + black))[И. А. Скребцов],
        [(группа)],
        [(подпись)],
        [(И. О. Фамилия)],

        text(style: "normal")[*Специальность*],
        grid.cell(colspan: 2, stroke: (bottom: 0.5pt + black))[
            09.02.08 Интеллектуальные интегрированные системы
        ],
        [],
        grid.cell(colspan: 2)[(шифр и наименование специальности)],
    )
]
#v(3em)
#[
    #set text(style: "italic")
    #grid(
        columns: (1fr, 1fr, 1fr),
        rows: auto,
        row-gutter: 0.5em,
        column-gutter: 1em,
        grid.cell(colspan: 2)[
            #set align(left)
            *Работа допущена к защите*
        ],
        [],
        text(style: "normal")[
            #set align(left)
            *Председатель ПЦК*
        ],
        grid.cell(stroke: (bottom: 0.5pt + black))[],
        grid.cell(stroke: (bottom: 0.5pt + black))[Т. А. Волкова],

        [],
        [(подпись)],
        [(И. О. Фамилия)],

        text(style: "normal")[
            #set align(left)
            *Зав. отделением*
        ],
        grid.cell(stroke: (bottom: 0.5pt + black))[],
        grid.cell(stroke: (bottom: 0.5pt + black))[О. В. Бондарук],

        [],
        [(подпись)],
        [(И. О. Фамилия)],
    )
]



#v(1fr)
Санкт-Петербург

2026
