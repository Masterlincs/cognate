<script>
  import '#lib/fonts.js';
  import { asset, resolve } from '$app/paths';
  import { goto } from '$app/navigation';
  import BarChart from '#lib/BarChart.svelte';
  import BeforeAfter from '#lib/BeforeAfter.svelte';
  import DotPlot from '#lib/DotPlot.svelte';
  import Rules from '#lib/Rules.svelte';
  import Search from '#lib/Search.svelte';
  import { loadRelease, setRoot } from '#lib/data.js';
  import { delta, fixed } from '#lib/format.js';
  import m from '#lib/generated/metrics.json';

  const REPO = 'https://github.com/Masterlincs/mtg-embeddings';
  const n = (x) => x.toLocaleString('en');
  const dep = m.deployed;
  const train = dep?.training;
  const enc = dep?.encoder;
  const d18 = m.promotion.d18;
  const x1 = m.promotion.x1;
  const sealed = m.sealed;
  const exam = m.exam;
  const exPre = exam['pretrained-e5-base-v2'];
  const exDep = dep ? exam[dep.run_id] : null;
  const margin = (rec, id) => rec?.failed.find((f) => f.id === id)?.margin ?? null;
  const fineTuned = m.dev['fine-tuned E5'];
  // the dev row whose runs include the deployed run (falls back to the first E5 recipe)
  const depRow = Object.values(m.dev).find((r) => r.runs.includes(dep?.run_id)) ?? fineTuned;
  const lexical = m.dev['keyword baseline (character TF-IDF)'];
  const pretrained = m.dev['pretrained E5-base'];
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const devRows = Object.entries(m.dev).map(([label, r]) => ({ label, values: { full: r.full, hard_lexical: r.hard_lexical }, highlight: r === depRow, note: r.runs.length > 1 ? `mean of ${r.runs.length} seeds, range shown` : '' }));
  const stepRows = Object.entries(m.steps).map(([s, v]) => ({ label: `seed ${s}`, a: v['2000'], b: v['4000'] }));
  const split = m.data.split_counts;

  // hero search: card names load only when the reader focuses the box
  let names = $state.raw(null);
  let ids = null;
  let loading = null;
  function loadNames() {
    loading ??= (async () => {
      setRoot(asset(''));
      const r = await loadRelease(fetch);
      names = r.names;
      ids = await r.idsReady;
    })().catch(() => { loading = null; });
    return loading;
  }
  async function open(i) {
    await loadNames();
    if (ids) goto(`${resolve('/')}?card=${ids[i]}`);
  }
</script>

<svelte:head>
  <title>How it works · Card Atlas</title>
  <meta name="description" content="How Card Atlas was built: a sentence-embedding model fine-tuned to find Magic cards with similar effects, how it was tested, and where it still fails." />
</svelte:head>

<article class="study">
  <header class="hero">
    <h1>Finding Magic cards by what they do, not what they say</h1>
    <p class="lede">Card Atlas ranks {n(m.data.split_counts.fit + m.data.split_counts.dev)} Magic: The Gathering cards by how similar their effects are. Under it is a small open text-embedding model, e5-base-v2, fine-tuned on community tags so that cards doing the same job end up close together even when their wording differs. This page covers the problem, the data, the training, how the model was tested and where it still goes wrong.</p>
    <div class="hero-actions">
      <div class="hero-search"><Search {names} id="hero-search" label="Open a card in the explorer" placeholder="Try a card, e.g. Counterspell" onfocus={loadNames} onpick={open} /></div>
      <a class="btn primary" href={resolve('/')}>Open the explorer</a>
    </div>
    <dl class="facts">
      <div><dt>Cards in the explorer</dt><dd class="num">{n(split.fit + split.dev)}</dd></div>
      <div><dt>Effect tags used as labels</dt><dd class="num">{n(m.data.tags_functional)}</dd></div>
      <div><dt>Dev score, fine-tuned</dt><dd class="num">{fixed(mean(depRow.full))}</dd></div>
      <div><dt>Dev score, before tuning</dt><dd class="num">{fixed(pretrained.full[0])}</dd></div>
    </dl>
  </header>

  <section>
    <h2>Keyword search matches words, not effects</h2>
    <p>Magic has over thirty thousand distinct cards, and players often want "something that does what this card does": a cheaper copy, the same effect in another colour, or a card for a deck theme. Search engines match words. These two cards share almost every word and do opposite things:</p>
    <div class="pair">
      <figure class="quote"><figcaption>Ghosts of the Innocent</figcaption><p>If a source would deal damage to a permanent or player, it deals <mark>half</mark> that damage, rounded down, to that permanent or player instead.</p></figure>
      <figure class="quote"><figcaption>Furnace of Rath</figcaption><p>If a source would deal damage to a permanent or player, it deals <mark>double</mark> that damage to that permanent or player instead.</p></figure>
      <figure class="quote good"><figcaption>Kitsune Palliator <span>a real match, different words</span></figcaption><p><Rules text={'{T}'} />: Prevent the next 1 damage that would be dealt to each creature and each player this turn.</p></figure>
    </div>
    <p>A model that reads meaning should place Ghosts next to damage-prevention cards such as Kitsune Palliator, not next to Furnace. Word overlap is still a strong signal: on the development set below, a plain keyword baseline (character TF-IDF) scores {fixed(lexical.full[0])}, above the untuned embedding model at {fixed(pretrained.full[0])}.</p>
  </section>

  <section>
    <h2>Community tags as a stand-in for "same effect"</h2>
    <div class="cols2">
      <div>
        <p>Card text comes from Scryfall's bulk data (snapshot {dep?.release_id.slice(0, 10) ?? ''}). Reminder text in parentheses is removed. The labels come from <a href="https://tagger.scryfall.com" rel="noopener">Scryfall Tagger</a>, where players tag cards with what they do ("removal", "card draw", "change target"). After dropping tags about cycles, names, errata and printing or set trivia, {n(m.data.tags_functional)} effect tags remain, with {n(m.data.functional_memberships)} card–tag links.</p>
        <p>Two cards count as similar when they share tags, weighted by how rare each tag is. That makes the tags a <strong>proxy</strong>: they reward any shared tag, including incidental ones, and the model is trained and scored on the same source. Every score on this page is a proxy score.</p>
      </div>
      <figure class="splits">
        <figcaption>How the {n(m.data.n_cards)} cards were split</figcaption>
        <div class="splitbar" role="img" aria-label="Fit {n(split.fit)}, development {n(split.dev)}, sealed test {n(split.sealed)} cards">
          <span class="fit" style:flex={split.fit}>fit {n(split.fit)}</span>
          <span class="dev" style:flex={split.dev}>dev</span>
          <span class="sealed" style:flex={split.sealed}>sealed</span>
        </div>
        <ul>
          <li><b>Fit ({n(split.fit)})</b>: training pairs come only from these.</li>
          <li><b>Dev ({n(split.dev)})</b>: used to compare runs and pick the recipe.</li>
          <li><b>Sealed ({n(split.sealed)})</b>: locked away and scored once, after the model was chosen.</li>
        </ul>
        <p class="muted small">Reprints and cards with identical text ({n(m.data.families)} families) stay in one split, so a model cannot score by remembering a copy.</p>
      </figure>
    </div>
  </section>

  <section>
    <h2>Contrastive fine-tuning of a small encoder</h2>
    <div class="flow" aria-label="Encoding pipeline">
      <div class="box"><small>Input</small>"{enc?.prefix ?? ''}" + rules text<br /><span class="muted">up to {enc?.max_length} tokens</span></div>
      <span class="arrow" aria-hidden="true">→</span>
      <div class="box accent"><small>Encoder</small>{enc?.model_id ?? 'e5-base-v2'}<br /><span class="muted">{enc?.pooling} pooling</span></div>
      <span class="arrow" aria-hidden="true">→</span>
      <div class="box"><small>Output</small>{enc?.dim}-number vector<br /><span class="muted">compared by cosine</span></div>
    </div>
    <div class="cols2">
      <div>
        <p>Each card becomes one vector, and neighbours are the cards with the most similar vectors. The pretrained model already knows English, but not which Magic effects are alike. Fine-tuning teaches it with pairs of cards that share an effect tag.</p>
        <p>Training uses a contrastive loss ({train?.objective === 'masked_infonce' ? 'InfoNCE' : train?.objective}): in each batch of {train?.batch_pairs} pairs, the model is pushed to score a card's partner above every other card in the batch. Other cards in the batch that also share a tag or a reprint family with it are masked out, so related cards are not pushed apart. Each card also gets {train?.hard_negatives} <em>hard negative</em>: a card with very similar wording but no shared tag, the Furnace-of-Rath case, chosen from its closest keyword matches.</p>
        <p class="muted small">Deployed run: {train?.total_updates} updates, learning rate {train?.lr?.toExponential()}, seed {train?.seed} (<code>{train?.source}</code>).</p>
      </div>
      <figure class="triplet" aria-label="Training example">
        <div class="t anchor"><small>Card</small>Ghosts of the Innocent</div>
        <div class="t pos"><small>Pulled closer: shares a tag</small>Kitsune Palliator</div>
        <div class="t neg"><small>Pushed away: similar words, no shared tag</small>Furnace of Rath</div>
      </figure>
    </div>
  </section>

  <section class="wide">
    <h2>The same card, before and after fine-tuning</h2>
    <p>Top {8} neighbours from the pretrained model (left) and the deployed fine-tuned model (right). Cards in both lists are marked. Pick an example or type any card.</p>
    <BeforeAfter showcase={m.showcase} />
  </section>

  <section class="wide">
    <h2>How much better, and how sure</h2>
    <p>The main score is nDCG@10 on the {n(split.dev)} development cards: for each card, how good its top 10 neighbours are by shared tags, with higher ranks counting more (1.0 is perfect). The <b>hard-lexical</b> view removes reprints and near-identical wording from the candidates, so a model cannot score by finding copies.</p>
    <BarChart rows={devRows} series={[{ key: 'full', label: 'full' }, { key: 'hard_lexical', label: 'hard-lexical' }]} caption="Dev tag-proxy nDCG@10 (higher is better). Trained models were run with 3 seeds; whiskers show the seed range." />

    <div class="cards3">
      <div class="stat">
        <h3>Promotion test</h3>
        <p class="big num">{delta(d18.contrasts.pretrained_e5.full.point)}</p>
        <p>over pretrained E5 (full view), adjusted lower bound {delta(d18.contrasts.pretrained_e5.full.lower_adjusted)}; {delta(d18.contrasts.trained_bge.full.point)} over a fine-tuned BGE-small. Cluster bootstrap over card families, {d18.m_contrasts} contrasts with a multiple-comparison correction. Decision: {d18.decision} ({d18.qualifier}).</p>
      </div>
      <div class="stat">
        <h3>Sealed test, run once</h3>
        <p class="big num">{delta(sealed.contrasts.pretrained_e5.full.point)}</p>
        <p>over pretrained E5 on the sealed cards, lower bound {delta(sealed.contrasts.pretrained_e5.full.lower_adjusted)}. Scored once, after the decision; the gate was {sealed.sealed_gate}. Model scored: {sealed.deployment.run_id}, {fixed(sealed.deployment.full)} full, {fixed(sealed.deployment.hard_lexical)} hard-lexical. {#if dep && sealed.deployment.run_id !== dep.run_id}The model deployed now was promoted later, on the development cards only, and has not been sealed-tested.{/if}</p>
      </div>
      <div class="stat">
        <h3>Trap exam</h3>
        <p class="big num">{exPre?.passed} → {exDep?.passed}<span class="of">/{exDep?.n}</span></p>
        <p>{exDep?.n} reviewed items, each a card, its true match and a lookalike that must rank lower (shared rider clauses, one-word flips, changed numbers). Pretrained E5 passes {exPre?.passed}; the deployed model passes {exDep?.passed}.</p>
      </div>
    </div>

    {#if stepRows.length}
      <h3 class="sub">Training longer: 2000 vs 4000 updates</h3>
      <p>Doubling the training length raised the dev score on every seed tried. {#if x1}A second promotion test, the same design with the 2000-update recipe added as a third comparator ({x1.m_contrasts} contrasts), decided: <b>{x1.decision}</b> ({x1.qualifier}). Against the 2000-update recipe the gain is {delta(x1.contrasts.incumbent_e5.full.point)} full (adjusted lower bound {delta(x1.contrasts.incumbent_e5.full.lower_adjusted)}) and {delta(x1.contrasts.incumbent_e5.hard_lexical.point)} hard-lexical. {#if dep?.run_id === x1.deployment?.run_id}The deployed model is now the 4000-update recipe, seed {x1.deployment.seed}.{/if}{:else}A formal promotion test of the 4000-update recipe needs a third seed, which is still training, so the deployed model is the 2000-update one.{/if}</p>
      <DotPlot rows={stepRows} aLabel="2000 updates" bLabel="4000 updates" caption="Dev tag-proxy nDCG@10 (full view) at the end of training, same seed and data order." />
    {/if}
  </section>

  <section>
    <h2>Where it still goes wrong</h2>
    <div class="limit">
      <h3>Deflecting Swat: the labels favour the wrong neighbour</h3>
      <p>Deflecting Swat redirects a spell, and it is free if you control a commander. The ideal neighbour is Redirect. The fine-tuned model ranks Obscuring Haze above it, which shares only the free-if-commander line. The tags explain it: Swat shares three tags with Obscuring Haze and two with Redirect, so training on tags pulls the wrong card closer. On the trap exam the margin went from {margin(exPre, 'T001') !== null ? fixed(margin(exPre, 'T001')) : '—'} before tuning to {margin(exDep, 'T001') !== null ? fixed(margin(exDep, 'T001')) : '—'} after (below zero is a fail). Fixing it means changing the labels, not the model. In the explorer, picking Swat's redirect line under "similar cards" does find Redirect.</p>
      <p><a href="{resolve('/')}?card=ae120613-97d6-4393-b39d-c3e6c076f5d6">See Deflecting Swat in the explorer</a></p>
    </div>
    <div class="limit">
      <h3>Lexical twins still fool it</h3>
      <p>Ghosts of the Innocent still ranks Furnace of Rath above Kitsune Palliator (trap-exam margin {margin(exDep, 'T021') !== null ? fixed(margin(exDep, 'T021')) : '—'}). A run with three hard negatives per card instead of one is testing whether more contrast against lookalikes helps; its result is not in yet.</p>
    </div>
    <ul class="plain">
      <li><b>Proxy, not people.</b> The ranking scores and promotion tests measure agreement with community tags. The trap exam is a set of 51 hand-made card comparisons, accepted by the author rather than a panel. No human relevance study has been run yet.</li>
      <li><b>Line matching is exploratory.</b> The explorer's Line match ranking scored below the whole-card ranking on an earlier model's tag benchmark, and "match one line" has not been evaluated.</li>
      <li><b>Snapshot.</b> Cards printed after the data snapshot are missing, and only English rules text is used.</li>
    </ul>
  </section>

  <section class="built">
    <h2>Built with</h2>
    <p>Python, PyTorch and Hugging Face Transformers for training and evaluation; scikit-learn for the map (t-SNE); SvelteKit for this static site, with no tracking and no server. Every result on this page is read from the run and evaluation records in the repository by a script.</p>
    <p class="links"><a class="btn" href={REPO} rel="noopener">Code on GitHub</a> <a class="btn" href="{REPO}/blob/master/WRITE_UP.MD" rel="noopener">Full write-up</a> <a class="btn primary" href={resolve('/')}>Open the explorer</a></p>
  </section>
</article>

<style>
  .study { max-width: 1080px; margin: 0 auto; padding: 0 clamp(16px, 4vw, 32px) 48px; }
  section, .hero { max-width: 760px; margin: 0 auto; padding: 56px 0 8px; }
  section.wide { max-width: none; }
  section.wide > p { max-width: 760px; }
  .hero { padding-top: clamp(32px, 8vh, 88px); }
  .hero h1 { margin: 12px 0 18px; }
  .lede { font-size: 1.12rem; color: var(--muted); }
  .hero-actions { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 26px 0 8px; }
  .hero-search { flex: 1 1 280px; }
  .hero-actions .btn { height: 48px; padding: 0 20px; font-size: 0.95rem; }
  .facts { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 30px 0 0; }
  .facts div { padding: 12px 14px; border-top: 2px solid var(--ink); }
  .facts dt { font-size: 0.76rem; color: var(--muted); }
  .facts dd { margin: 2px 0 0; font-size: 1.35rem; }
  h2 { margin: 6px 0 16px; }
  .sub { margin: 36px 0 8px; }
  .small { font-size: 0.84rem; }
  .pair { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }
  .quote { margin: 0; padding: 14px 16px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); font-size: 0.9rem; }
  .quote figcaption { font: 600 1rem var(--sans); margin-bottom: 6px; }
  .quote figcaption span { display: block; font: 400 0.74rem var(--sans); color: var(--muted); }
  .quote p { margin: 0; font-family: var(--serif); font-size: 0.95rem; }
  .quote.good { border-color: color-mix(in srgb, var(--wubrg-g) 55%, var(--line)); }
  mark { background: var(--mark); color: inherit; padding: 0 2px; }
  .cols2 { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 28px; align-items: start; }
  .splits { margin: 0; padding: 16px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); font-size: 0.88rem; }
  .splits figcaption { font-weight: 600; margin-bottom: 10px; }
  .splitbar { display: flex; height: 30px; border-radius: 4px; overflow: hidden; font: 0.72rem/30px var(--sans); color: #fff; }
  .splitbar span { padding: 0 6px; white-space: nowrap; overflow: hidden; }
  .fit { background: color-mix(in srgb, var(--wubrg-u) 70%, #000); }
  .dev { background: color-mix(in srgb, var(--wubrg-g) 70%, #000); }
  .sealed { background: color-mix(in srgb, var(--wubrg-b) 70%, #000); }
  .splits ul { margin: 12px 0 8px; padding-left: 1.1em; }
  .splits li { margin: 3px 0; }
  .flow { display: flex; align-items: stretch; gap: 10px; margin: 18px 0 24px; }
  .box { flex: 1; padding: 12px 14px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); font-size: 0.88rem; }
  .box small, .t small { display: block; font: 600 0.68rem var(--sans); color: var(--muted); margin-bottom: 4px; }
  .box.accent { border-color: var(--ink); }
  .arrow { align-self: center; color: var(--muted); font-size: 1.3rem; }
  .triplet { margin: 0; display: grid; gap: 10px; }
  .t { padding: 12px 14px; border-radius: var(--radius); border: 1px solid var(--line); background: var(--surface); font: 600 0.98rem var(--sans); }
  .t.anchor { border-color: var(--ink); }
  .t.pos { border-left: 4px solid var(--wubrg-g); margin-left: 24px; }
  .t.neg { border-left: 4px solid var(--wubrg-r); margin-left: 24px; }
  .cards3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 32px 0 8px; }
  .stat { padding-top: 12px; border-top: 2px solid var(--ink); font-size: 0.86rem; }
  .stat h3 { font-size: 0.95rem; }
  .stat p { margin: 0 0 6px; }
  .big { font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1; margin: 6px 0 8px !important; }
  .of { font-size: 1rem; color: var(--muted); }
  .limit { margin: 18px 0 26px; }
  .limit h3 { margin-bottom: 6px; }
  .plain { padding-left: 1.1em; }
  .plain li { margin: 6px 0; }
  .built .links { display: flex; flex-wrap: wrap; gap: 8px; }
  @media (max-width: 760px) {
    .facts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .pair, .cols2, .cards3 { grid-template-columns: minmax(0, 1fr); }
    .flow { flex-direction: column; }
    .arrow { transform: rotate(90deg); }
  }
</style>
