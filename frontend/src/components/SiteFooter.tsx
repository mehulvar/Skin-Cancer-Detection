const SOURCES = [
  { label: "HAM10000 (Tschandl et al., 2018)", href: "https://doi.org/10.7910/DVN/DBW86T" },
  { label: "ISIC 2018 Challenge", href: "https://challenge.isic-archive.com/data/#2018" },
  { label: "SCIN dataset (Google Research)", href: "https://github.com/google-research-datasets/scin" },
  { label: "Imagenette (fast.ai)", href: "https://github.com/fastai/imagenette" },
  { label: "Describable Textures Dataset", href: "https://www.robots.ox.ac.uk/~vgg/data/dtd/" },
];

export function SiteFooter() {
  return (
    <footer className="mt-auto">
      <section className="bg-linen">
        <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-14 md:flex-row md:items-center md:justify-between md:px-8">
          <div className="max-w-2xl">
            <p className="font-serif text-3xl text-ink">Not a diagnosis.</p>
            <p className="mt-3 text-sm leading-relaxed text-muted">
              SkinSight is an educational research project. It can be wrong, it has not been clinically validated,
              and it should never replace an examination by a doctor. If a spot is new, changing, bleeding or
              worrying you, see a GP or dermatologist.
            </p>
          </div>
          <a
            href="#analyse"
            className="shrink-0 self-start rounded-full bg-ink px-7 py-3.5 text-sm font-medium text-cream transition hover:bg-ink-soft md:self-auto"
          >
            Begin an analysis
          </a>
        </div>
      </section>
      <div className="mx-auto grid max-w-6xl gap-8 px-4 py-12 text-sm md:grid-cols-3 md:px-8">
        <div>
          <p className="font-serif text-2xl text-ink">SkinSight</p>
          <p className="mt-2 text-muted">Skin lesion classification research project by Mehul Varshney.</p>
        </div>
        <div>
          <p className="eyebrow">Data sources</p>
          <ul className="mt-3 flex flex-col gap-1.5">
            {SOURCES.map((s) => (
              <li key={s.href}>
                <a href={s.href} target="_blank" rel="noreferrer" className="text-muted underline-offset-4 hover:text-ink hover:underline">
                  {s.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="eyebrow">Code</p>
          <a
            href="https://github.com/mehulvar/Skin-Cancer-Detection"
            target="_blank"
            rel="noreferrer"
            className="mt-3 inline-block text-muted underline-offset-4 hover:text-ink hover:underline"
          >
            github.com/mehulvar/Skin-Cancer-Detection
          </a>
          <p className="mt-6 text-xs text-muted">
            Images are used for non-commercial research and education under their respective licences.
          </p>
        </div>
      </div>
    </footer>
  );
}
