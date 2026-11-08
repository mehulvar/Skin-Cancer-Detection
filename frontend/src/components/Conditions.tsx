import { CLASSES } from "@/lib/classes";
import { SAMPLES } from "@/lib/content";
import { CategoryBadge } from "./ResultPanel";

export function Conditions() {
  return (
    <section id="conditions" className="py-20 md:py-28">
      <div className="mx-auto max-w-6xl px-4 md:px-8">
        <div className="grid gap-6 md:grid-cols-2 md:items-end">
          <div>
            <p className="eyebrow">Conditions</p>
            <h2 className="mt-3 font-serif text-4xl leading-tight text-ink md:text-6xl">
              What the model can recognise.
            </h2>
          </div>
          <p className="text-muted md:pb-2">
            Seven lesion types from HAM10000 — two cancers, one precancer and four benign lesions — plus two
            safety classes that keep the model honest about what it&apos;s looking at.
          </p>
        </div>

        <ul className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {CLASSES.map((c) => {
            const example = SAMPLES.find((s) => s.label === c.key);
            return (
              <li key={c.key} className="flex flex-col overflow-hidden rounded-[1.75rem] border border-line bg-cream">
                <div className="aspect-[16/10] overflow-hidden bg-linen">
                  {example && (
                    // eslint-disable-next-line @next/next/no-img-element -- local sample thumbnail
                    <img src={example.src} alt={`Example: ${c.short}`} loading="lazy" className="h-full w-full object-cover" />
                  )}
                </div>
                <div className="flex flex-1 flex-col gap-3 p-6">
                  <CategoryBadge category={c.category} />
                  <h3 className="font-serif text-2xl text-ink">{c.name}</h3>
                  <p className="text-sm leading-relaxed text-muted">{c.description}</p>
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
