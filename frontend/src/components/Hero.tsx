import { METRICS, SAMPLES, percent } from "@/lib/content";

const HERO_LABELS = ["mel", "nv", "bcc"] as const;

export function Hero() {
  const images = HERO_LABELS.map((label) => SAMPLES.find((s) => s.label === label)).filter(
    (s): s is NonNullable<typeof s> => Boolean(s),
  );

  return (
    <section id="top" className="relative overflow-hidden">
      <div className="mx-auto grid max-w-6xl items-center gap-12 px-4 pb-20 pt-14 md:grid-cols-[1.1fr_1fr] md:px-8 md:pb-28 md:pt-20">
        <div className="rise">
          <p className="eyebrow">AI-assisted skin lesion analysis</p>
          <h1 className="mt-5 font-serif text-5xl leading-[1.02] text-ink sm:text-6xl md:text-7xl">
            Clarity for every <em className="text-clay-deep">mark</em> on your skin.
          </h1>
          <p className="mt-6 max-w-lg text-lg leading-relaxed text-muted">
            A deep-learning model trained on more than 10,000 dermoscopic images recognises seven types of skin
            lesion — from harmless moles to melanoma — and knows when an image isn&apos;t a lesion at all.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a href="#analyse" className="rounded-full bg-ink px-7 py-3.5 text-sm font-medium text-cream transition hover:bg-ink-soft">
              Analyse an image
            </a>
            <a href="#samples" className="rounded-full border border-ink/25 px-7 py-3.5 text-sm font-medium text-ink transition hover:border-ink">
              Try a sample
            </a>
          </div>
          {METRICS && (
            <dl className="mt-12 grid max-w-md grid-cols-3 gap-6 border-t border-line pt-6">
              <div>
                <dt className="text-xs text-muted">Test accuracy</dt>
                <dd className="mt-1 font-serif text-3xl text-ink">{percent(METRICS.accuracy, 0)}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted">Cancer sensitivity</dt>
                <dd className="mt-1 font-serif text-3xl text-ink">{percent(METRICS.dangerSensitivity, 0)}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted">Runs on</dt>
                <dd className="mt-1 font-serif text-3xl text-ink">Device</dd>
              </div>
            </dl>
          )}
        </div>

        <div className="relative mx-auto grid w-full max-w-xs grid-cols-2 gap-3 md:max-w-md md:gap-4" aria-hidden>
          <div className="arch col-span-1 row-span-2 aspect-[3/5] overflow-hidden bg-linen">
            {images[0] && (
              // eslint-disable-next-line @next/next/no-img-element -- local dermoscopy sample
              <img src={images[0].src} alt="" className="h-full w-full object-cover" />
            )}
          </div>
          <div className="aspect-square overflow-hidden rounded-[1.5rem] bg-linen">
            {images[1] && (
              // eslint-disable-next-line @next/next/no-img-element -- local dermoscopy sample
              <img src={images[1].src} alt="" className="h-full w-full object-cover" />
            )}
          </div>
          <div className="aspect-square overflow-hidden rounded-full bg-linen">
            {images[2] && (
              // eslint-disable-next-line @next/next/no-img-element -- local dermoscopy sample
              <img src={images[2].src} alt="" className="h-full w-full object-cover" />
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
