const FEATURES = [
  {
    title: "Trained on real dermoscopy",
    body: "10,015 expert-labelled images from HAM10000 plus the ISIC Archive, covering the seven most common pigmented lesion types.",
  },
  {
    title: "Knows what it doesn't know",
    body: "Extra classes for rashes and other non-cancerous skin conditions, and for images that aren't skin at all — so a photo of a dog doesn't get called a melanoma.",
  },
  {
    title: "Private by design",
    body: "The model runs in your browser with WebAssembly. Images are never uploaded, stored or seen by anyone else.",
  },
  {
    title: "Measured honestly",
    body: "Scored on the official ISIC 2018 test set — images from different lesions than training, so the numbers reflect real-world generalisation.",
  },
];

export function Features() {
  return (
    <section className="py-20 md:py-28">
      <div className="mx-auto max-w-6xl px-4 md:px-8">
        <p className="eyebrow">Why SkinSight</p>
        <h2 className="mt-3 max-w-2xl font-serif text-4xl leading-tight text-ink md:text-6xl">
          A second look, built on evidence.
        </h2>
        <ol className="mt-14 grid gap-px overflow-hidden rounded-[2rem] border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map((f, i) => (
            <li key={f.title} className="flex flex-col gap-4 bg-cream p-7">
              <span className="font-serif text-5xl text-clay">{i + 1}</span>
              <h3 className="text-lg font-medium text-ink">{f.title}</h3>
              <p className="text-sm leading-relaxed text-muted">{f.body}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
