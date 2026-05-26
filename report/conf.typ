#let template(body) = {
    set document(
        author: "Иван Скребцов",
        title: "Дипломная работа",
    )

    set page(
        paper: "a4",
        footer: context {
            set text(size: 12pt, fill: gray)
            h(1fr)
            counter(page).display("1")
        },
    )

    set par(
        first-line-indent: (amount: 1.7cm, all: true),
        justify: true,
        leading: 0.75em,
        spacing: 0.75em,
    )

    set text(
        font: "Times New Roman",
        size: 14pt,
        top-edge: "ascender",
        bottom-edge: "descender",
        lang: "ru",
    )

    // Headings
    set heading(

      numbering: (..nums) => {
        if nums.pos().len() == 1 {
          none
        } else {
          "1."
        }
      }
      )

    show heading: set text(size: 14pt, weight: "regular")
    show heading: block.with(above: 12pt, below: 6pt)
    show heading.where(level: 1): it => block(width: 100%)[
        #set align(center)
        #upper()[#it]
    ]
    show table: set text(size: 12pt)

    // Outline
    show outline.entry: it => {
        link(
            it.element.location(),
            it.indented(it.prefix(), [#it.body() #h(1fr) #it.page()]),
        )
    }

    // Lists
    set list(
        marker: [•],
        indent: 1.7cm,
    )

    // Code
    show raw.where(block: false): body => {
        box(
            fill: luma(245),
            inset: (x: 3pt, y: 0pt),
            outset: (y: 3pt),
            radius: 2pt,
        )[
            #set text(font: "Courier New")
            #body
        ]
    }
    show raw.where(block: true): body => {
        set align(center)
        block(fill: luma(245), radius: 2pt, width: 90%, inset: (
            x: 12pt,
            y: 8pt,
        ))[
            #set align(left)
            #set text(font: "Courier New")
            #body
        ]
    }

    body
}
