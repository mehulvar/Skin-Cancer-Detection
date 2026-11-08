import { Analyzer } from "@/components/Analyzer";
import { Conditions } from "@/components/Conditions";
import { Features } from "@/components/Features";
import { Hero } from "@/components/Hero";
import { Method } from "@/components/Method";
import { Performance } from "@/components/Performance";
import { SiteFooter } from "@/components/SiteFooter";
import { SiteHeader } from "@/components/SiteHeader";

export default function Home() {
  return (
    <>
      <SiteHeader />
      <main className="flex-1">
        <Hero />
        <Analyzer />
        <Features />
        <Conditions />
        <Performance />
        <Method />
      </main>
      <SiteFooter />
    </>
  );
}
