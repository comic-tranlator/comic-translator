#import "conf.typ": template

#show: template

#include "sections/title.typ"
#pagebreak()

#include "sections/properties.typ"
#pagebreak()

#include "sections/plan.typ"
#pagebreak()

#include "sections/annotation.typ"
#pagebreak()

#heading(numbering: none)[Содержание]
#outline(
    title: none,
    depth: 2,
)

#pagebreak()

#include "sections/intro.typ"
#pagebreak()

#include "sections/parts/theoretical.typ"
#pagebreak()

#include "sections/parts/implementation.typ"
#pagebreak()

#include "sections/parts/economical.typ"
#pagebreak()

#include "sections/parts/safety.typ"
#pagebreak()

#include "sections/conclusion.typ"
#pagebreak()

#include "sections/sources.typ"
#pagebreak()

#include "sections/application.typ"
