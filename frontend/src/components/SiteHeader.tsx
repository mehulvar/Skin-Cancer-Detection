const NAV = [
  { href: "#analyse", label: "Analyse" },
  { href: "#samples", label: "Samples" },
  { href: "#conditions", label: "Conditions" },
  { href: "#performance", label: "Performance" },
  { href: "#method", label: "Method" },
];

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-line/70 bg-cream/85 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 md:px-8">
        <a href="#top" className="font-serif text-2xl tracking-tight text-ink">
          SkinSight
        </a>
        <nav aria-label="Primary" className="hidden gap-8 text-sm text-text md:flex">
          {NAV.map((item) => (
            <a key={item.href} href={item.href} className="transition hover:text-clay-deep">
              {item.label}
            </a>
          ))}
        </nav>
        <a
          href="#analyse"
          className="rounded-full bg-ink px-4 py-2 text-sm font-medium text-cream transition hover:bg-ink-soft md:px-5"
        >
          Analyse an image
        </a>
      </div>
    </header>
  );
}
