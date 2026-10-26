<script>
  import '#lib/fonts.js';
  import { asset, resolve } from '$app/paths';
  import { goto } from '$app/navigation';
  import BarChart from '#lib/BarChart.svelte';
  import BeforeAfter from '#lib/BeforeAfter.svelte';
  import ContrastPlot from '#lib/ContrastPlot.svelte';
  import DotPlot from '#lib/DotPlot.svelte';
  import Rules from '#lib/Rules.svelte';
  import Search from '#lib/Search.svelte';
  import { loadRelease, setRoot } from '#lib/data.js';
  import { delta, fixed, mean } from '#lib/format.js';
  import { contrastRows, runScore } from '#lib/study.js';
  import { decisionNote } from '#lib/wording.js';
  import m from '#lib/generated/metrics.json';

  const REPO = 'https://github.com/Masterlincs/cognate';
  const n = (x) => x.toLocaleString('en');
  const dep = m.deployed;
  const train = dep?.training;
  const enc = dep?.encoder;
  const promo = m.promotion.x1 ?? m.promotion.d18;
  const sealed = m.sealed;
  const exam = m.exam;
  const exPre = exam['pretrained-e5-base-v2'];
  const exDep = dep ? exam[dep.run_id] : null;
  const margin = (rec, id) => rec?.failed.find((f) => f.id === id)?.margin ?? null;
  const marginText = (rec, id) => (margin(rec, id) === null ? '—' : fixed(margin(rec, id)));
  const fineTuned = m.dev['fine-tuned E5, 2000 updates'];
  // the dev row whose runs include the deployed run (falls back to the first E5 recipe)
  const depRow = Object.values(m.dev).find((r) => r.runs.includes(dep?.run_id)) ?? fineTuned;
  const depScore = runScore(depRow, dep?.run_id) ?? mean(depRow.full);
  const lexical = m.dev['character n-gram TF-IDF'];
  const pretrained = m.dev['pretrained E5-base'];
  const split = m.data.split_counts;
  const [stepA, stepB] = Object.keys(Object.values(m.steps)[0] ?? {});
  const stepRows = Object.entries(m.steps).map(([s, v]) => ({ label: `seed ${s}`, a: v[stepA], b: v[stepB] }));

  // evaluation sizes from runs/eval/<dev benchmark>/confirm-x1-s1/summary.json and benchmark_v3.json
  const DEV_ANCHORS = 3016;
  const DEV_UNIVERSE = 27573;
  const TRUNCATED = 260;
  const KEYWORD_ONLY = 1099;
  const seeds = Math.max(...Object.values(m.dev).map((r) => r.runs.length));
  const spread = Math.max(...Object.values(m.dev).flatMap((r) => [r.full, r.hard_lexical].map((xs) => Math.max(...xs) - Math.min(...xs))));
  const devRows = Object.entries(m.dev).map(([label, r]) => ({
    label,
    values: { full: r.full, hard_lexical: r.hard_lexical },
    highlight: r === depRow,
    note: [r === depRow && 'deployed recipe', r.runs.length > 1 && `mean of ${r.runs.length} seeds`].filter(Boolean).join(', '),
  }));

  const contrastLabels = { pretrained_e5: 'pretrained E5', trained_bge: 'fine-tuned BGE-small', incumbent_e5: `the ${stepA}-update recipe` };
  const promoIsDeployed = promo.deployment?.run_id === dep?.run_id;
  const sealedIsDeployed = sealed.deployment.run_id === dep?.run_id;
  const panels = [
    { title: `Development cards: ${promoIsDeployed ? 'the deployed recipe' : 'the promoted recipe'}`, note: `${seeds} seeds, ${promo.m_contrasts} comparisons corrected together. ${decisionNote(promo)}`, rows: contrastRows(promo, contrastLabels) },
    { title: `Sealed cards: ${sealedIsDeployed ? 'the deployed model' : `the earlier ${stepA}-update model`}, one test`, note: `${sealed.m_contrasts} comparisons. Gate ${sealed.sealed_gate}.`, rows: contrastRows(sealed, contrastLabels) },
  ];

  const steps = [
    ['data', 'Data', `${n(m.data.n_cards)} cards from Scryfall`],
    ['labels', 'Labels', `${n(m.data.tags_functional)} effect tags`],
    ['splits', 'Splits', 'fit, dev, sealed'],
    ['training', 'Training', `${n(train?.total_updates ?? 0)} updates`],
    ['results', 'Results', `${fixed(depScore)} dev nDCG@10`],
    ['limits', 'Where it fails', exDep ? `${exDep.n - exDep.passed} of ${exDep.n} trap items` : 'known misses'],
    ['built', 'Built with', 'tools and credits'],
  ];
  const allClear = panels.every((p) => p.rows.every((r) => r.adj > 0));
  const stepNo = Object.fromEntries(steps.map(([id], i) => [id, i + 1]));

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
  <title>How it works · Cognate</title>
  <meta name="description" content="How I built Cognate, a site that finds Magic cards with similar effects using a fine-tuned embedding model. How I tested it, and where it still fails." />
  <meta property="og:title" content="How Cognate works" />
  <meta property="og:description" content="How I built Cognate, a site that finds Magic cards with similar effects using a fine-tuned embedding model. How I tested it, and where it still fails." />
  <meta property="og:url" content="https://cognate.lincolnvw.dev/how-it-works" />
</svelte:head>

{#snippet eyebrow(id)}<p class="eyebrow"><span class="num">{stepNo[id]}</span> {steps[stepNo[id] - 1][1]}</p>{/snippet}

<article class="study">
  <header class="hero">
    <h1>Finding Magic cards by what they do, not what they say</h1>
    <p class="lede">Pick a Magic: The Gathering card, and Cognate lists the cards its model rates as most similar in effect, often even when the words differ. The model is e5-base-v2, a small open text-embedding model that I fine-tuned on community tags. This page shows how I built and tested it, and where it still fails.</p>
    <div class="hero-actions">
      <div class="hero-search"><Search {names} id="hero-search" label="Open a card in the explorer" placeholder="Try a card, such as Counterspell" onfocus={loadNames} onpick={open} /></div>
      <a class="btn primary" href={resolve('/')}>Open the explorer</a>
    </div>
    <dl class="facts">
      <div><dt>Cards you can search</dt><dd class="num">{n(split.fit + split.dev)}</dd></div>
      <div><dt>Dev nDCG@10, TF-IDF baseline</dt><dd class="num">{fixed(lexical.full[0])}</dd></div>
      <div><dt>Dev nDCG@10, deployed</dt><dd class="num">{fixed(depScore)}</dd></div>
      <div><dt>Dev nDCG@10, untuned</dt><dd class="num">{fixed(pretrained.full[0])}</dd></div>
    </dl>
  </header>

  <aside class="short" aria-labelledby="short-title">
    <h2 id="short-title">In short</h2>
    <ul>
      <li>I fine-tuned a small text-embedding model so that cards with a similar effect get similar vectors. You pick a card, and the site shows the cards with the closest vectors.</li>
      <li>On the development cards, the tuned model scores {fixed(depScore)} nDCG@10. The untuned model scores {fixed(pretrained.full[0])}. A word-matching baseline scores {fixed(lexical.full[0])}.</li>
      <li>The answer key is community tags, and the model trained on the same kind of tags. No person has judged the results.</li>
      <li>On six unseen cards the results are mixed. Tuning clearly helped one, changed little or gave mixed lists on three, and dropped a close match on two (see Results).</li>
    </ul>
  </aside>

  <nav class="pipeline" aria-label="On this page">
    <ol>
      {#each steps as [id, title, detail], i (id)}
        <li><a href="#{id}"><span class="no num" aria-hidden="true">{i + 1}</span><b>{title}</b><small>{detail}</small></a></li>
      {/each}
    </ol>
  </nav>

  <section id="problem">
    <h2>Keyword search matches words, not effects</h2>
    <p>Players often want "a card that does what this card does". It might be a cheaper copy, the same effect in another colour, or a card for a deck theme. Search engines match words. But the first two cards below share almost every word, and they do opposite things.</p>
    <div class="pair bleed">
      <figure class="quote"><figcaption>Ghosts of the Innocent</figcaption><p>If a source would deal damage to a permanent or player, it deals <mark>half</mark> that damage, rounded down, to that permanent or player instead.</p></figure>
      <figure class="quote"><figcaption>Furnace of Rath <span>same words, opposite effect</span></figcaption><p>If a source would deal damage to a permanent or player, it deals <mark>double</mark> that damage to that permanent or player instead.</p></figure>
      <figure class="quote good"><figcaption>Kitsune Palliator <span>a real match, different words</span></figcaption><p><Rules text={'{T}'} />: Prevent the next 1 damage that would be dealt to each creature and each player this turn.</p></figure>
    </div>
    <p>A model that reads for meaning should put Ghosts next to damage-prevention cards such as Kitsune Palliator, not next to Furnace. But shared words are still a strong signal. On the development cards, a baseline that only matches wording (character n-gram TF-IDF, no learning) scores {fixed(lexical.full[0])}. The untuned embedding model scores lower, at {fixed(pretrained.full[0])}.</p>
  </section>

  <section id="data">
    {@render eyebrow('data')}
    <h2>Rules text from Scryfall</h2>
    <p>The card text comes from Scryfall's bulk data, snapshot {dep?.release_id.slice(0, 10) ?? ''}. That gives {n(m.data.n_cards)} cards. I remove all text inside matched parentheses. Most of it is reminder text, but a few parentheticals matter to the rules, and they go too. For the {n(m.data.multi_face)} cards with more than one face, I join the faces into one text. I use only English rules text.</p>
  </section>

  <section id="labels">
    {@render eyebrow('labels')}
    <h2>Community tags stand in for "same effect"</h2>
    <p>The labels come from <a href="https://tagger.scryfall.com" rel="noopener">Scryfall Tagger</a>, where players tag cards with what they do, such as "removal", "card draw" and "change target". I dropped the tags about cycles, names, errata, and printing or set trivia. That leaves {n(m.data.tags_functional)} effect tags and {n(m.data.functional_memberships)} card–tag links.</p>
    <p>Two cards count as similar when they share tags, and rare tags count for more. This makes the tags a <strong>proxy</strong> for "same effect". They reward any shared tag, even an incidental one. Also, I train and score the model on the same source. The benchmark scores and promotion tests use this proxy, not a human judgement. The neighbour lists use the model's cosine similarity. The trap exam below uses matches that I accepted myself.</p>
  </section>

  <section id="splits">
    {@render eyebrow('splits')}
    <h2>Three splits, one held out for a single final test</h2>
    <figure class="splits">
      <div class="splitbar" role="img" aria-label="Fit {n(split.fit)}, development {n(split.dev)}, sealed {n(split.sealed)} cards">
        <span class="fit" style:flex={split.fit}></span>
        <span class="dev" style:flex={split.dev}></span>
        <span class="sealed" style:flex={split.sealed}></span>
      </div>
      <ul>
        <li><i class="fit"></i><b>Fit</b> <span class="num">{n(split.fit)}</span><span>Training pairs come only from these.</span></li>
        <li><i class="dev"></i><b>Dev</b> <span class="num">{n(split.dev)}</span><span>Used to compare runs and pick the recipe.</span></li>
        <li><i class="sealed"></i><b>Sealed</b> <span class="num">{n(split.sealed)}</span><span>Locked away. One authorised test used it, on the earlier 2000-update model. That test restarted once after a crash, before any model was scored. Not in the explorer.</span></li>
      </ul>
    </figure>
    <p>Cards with identical text, cards that differ only by their own name, and melded pairs form {n(m.data.families)} families. Most families hold one card. Each family stays in one split. Near-identical cards can still land in different splits, so the hard-lexical view below controls for that.</p>
  </section>

  <section id="training">
    {@render eyebrow('training')}
    <h2>Contrastive fine-tuning of a small encoder</h2>
    <ol class="flow" aria-label="How a card becomes a vector">
      <li><small>Input</small>"{enc?.prefix ?? ''}" + rules text<span>up to {enc?.max_length} tokens</span></li>
      <li class="accent"><small>Encoder</small>{enc?.model_id ?? 'e5-base-v2'}<span>{enc?.pooling} pooling</span></li>
      <li><small>Output</small>{enc?.dim}-number vector<span>compared by cosine</span></li>
    </ol>
    <p>Each card becomes one vector. A card's neighbours are the cards with the most similar vectors. The pretrained model already finds some related effects, but it scores below the wording baseline on this benchmark. Fine-tuning moves its rankings toward the community tags. It learns from pairs of cards that share an effect tag.</p>
    <div class="cols2">
      <p>The loss is contrastive ({train?.objective === 'masked_infonce' ? 'InfoNCE' : train?.objective}). Each batch has {train?.batch_pairs} pairs. The loss pushes the model to score a card's partner above every other card in the batch. Cards from one family never share a batch. The loss ignores any batch card that shares a training tag with the card, so related cards do not get pushed apart. (A training tag is one that 2 to 500 training cards carry.) Each pair also gets {train?.hard_negatives} <em>hard negative</em>: a card with similar wording but no shared tag. I pick it at random from the card's 30 closest training cards by character n-gram TF-IDF.</p>
      <figure class="triplet" aria-label="An illustration of the loss, not a real training example">
        <div class="t anchor"><small>Card</small>Ghosts of the Innocent</div>
        <div class="t pos"><small>Should rank close: a real match</small>Kitsune Palliator</div>
        <div class="t neg"><small>Should rank lower: similar words, different effect</small>Furnace of Rath</div>
        <figcaption class="muted small">Illustration only. Kitsune Palliator is a development card, so this pair is never used in training.</figcaption>
      </figure>
    </div>
    <p class="muted small">Deployed run: {n(train?.total_updates ?? 0)} updates, learning rate {train?.lr?.toExponential()}, seed {train?.seed} (<code>{train?.source}</code>).</p>
  </section>

  <section id="results">
    {@render eyebrow('results')}
    <h2>What changed, and how sure I can be</h2>

    <h3 class="sub">The same card, before and after</h3>
    <p>Each example shows the top 8 neighbours from the pretrained model and from the deployed fine-tuned model. The first six are familiar cards from the training split. They show behaviour, not performance on unseen cards. The other six are development cards, picked with a random seed (20261026) before I looked at their lists. I did not choose them, and I did not remove any. Six is too few to measure anything. They only give a rough feel for unseen cards: help on some, little change on others, and misses. You can also type the name of any card. The explorer hides {n(KEYWORD_ONLY)} keyword-only cards, but this view does not.</p>
    <div class="bleed"><BeforeAfter showcase={m.showcase} /></div>

    <h3 class="sub">Scores against baselines</h3>
    <p>The score is nDCG@10, averaged over {n(DEV_ANCHORS)} development cards ({n(split.dev)} cards in the split, minus those with no rules text or no tag seen in training). Each card is ranked against {n(DEV_UNIVERSE)} training and development cards that have rules text. The score rates the top 10 neighbours. A neighbour earns the summed weight of the tags it shares, and rare tags weigh more. Higher ranks count for more, and 1.0 is perfect. The <b>hard-lexical</b> view also removes cards from the same family and cards with at least 80% word overlap with the query. That way, a model cannot score points by finding copies.</p>
    <BarChart rows={devRows} series={[{ key: 'full', label: 'full' }, { key: 'hard_lexical', label: 'hard-lexical' }]} axisLabel="nDCG@10" caption="Development tag-proxy nDCG@10. Higher is better, and the bars start at zero. Each trained system shows the mean of {seeds} seeds, and the seed scores differ by at most {fixed(spread)}. The deployed model is seed {train?.seed} of its recipe. I fixed that seed before training. It is the lowest of the three seeds, with {fixed(depScore)} full." />

    <h3 class="sub">Is the gain more than noise?</h3>
    <p>I resample card families and the three seeds together, 12,000 times. That gives each difference a 95% interval and a one-sided lower bound. I adjust the lower bounds for the number of comparisons (a Bonferroni correction). With three seeds, the intervals mostly reflect which cards were sampled, not how much training varies. A model passes a comparison only if it meets both rules. In the full view, the gain must be at least 0.005 with an adjusted lower bound above 0. In the hard-lexical view, the gain must be at least 0 with an adjusted lower bound above −0.010. {#if allClear}Every adjusted lower bound is above zero.{/if}</p>
    <ContrastPlot {panels} caption="Difference in tag-proxy nDCG@10 between the fine-tuned E5 recipe (3-seed mean) and each comparator. The top panel is the 4000-update recipe and the bottom panel is the 2000-update recipe." />
    <p class="caveat">The development results are selection-exposed. Those cards helped decide which recipe and checkpoint to keep. The sealed numbers are for {sealedIsDeployed ? 'the deployed model' : `the earlier ${stepA}-update model (${sealed.deployment.run_id}: ${fixed(sealed.deployment.full)} full, ${fixed(sealed.deployment.hard_lexical)} hard-lexical)`}. {#if !sealedIsDeployed}The sealed split is now used up, so the deployed {stepB}-update model has no sealed result.{/if}</p>

    {#if stepRows.length}
      <h3 class="sub">Training longer: {stepA} vs {stepB} updates</h3>
      <p>Doubling the training length raised the dev score on every seed that I tried. {#if m.promotion.x1}Against the {stepA}-update recipe, the gain is {delta(m.promotion.x1.contrasts.incumbent_e5.full.point)} full and {delta(m.promotion.x1.contrasts.incumbent_e5.hard_lexical.point)} hard-lexical. The gain is small, and I measured it only on the development tag proxy.{/if}</p>
      <DotPlot rows={stepRows} aLabel="{stepA} updates" bLabel="{stepB} updates" caption="Development tag-proxy nDCG@10 (full view) at the end of training. Each pair of runs uses the same seed and data order. The axis is zoomed in and does not start at zero." />
    {/if}

    {#if exPre && exDep}
      <h3 class="sub">Trap exam</h3>
      <div class="exam">
        <div class="meters" role="img" aria-label="Pretrained E5 passes {exPre.passed} of {exPre.n}; the deployed model passes {exDep.passed} of {exDep.n}">
          {#each [['pretrained E5', exPre], ['deployed', exDep]] as [label, rec] (label)}
            <div class="meter"><span class="ml">{label}</span><span class="cells">{#each Array.from({ length: rec.n }, (_, i) => i) as i (i)}<i class:on={i < rec.passed}></i>{/each}</span><b class="num">{rec.passed}<span>/{rec.n}</span></b></div>
          {/each}
        </div>
        <p>The exam has {exDep.n} items. Two come from my own review of the app. Simple text rules in my code proposed the other 49, and I checked and accepted them. Each item names a card or line, the match I accept, and a lookalike that must rank lower. The lookalikes use shared rider clauses, one-word flips and changed numbers. The rules chose the matches and lookalikes from the same Tagger tags that the model trains on, so the exam is not independent of the training labels. Its fit cards also appear in the training pairs.</p>
        <p>The deployed model fails {exDep.failed.length} ({exDep.failed.map((f) => f.id).join(', ')}), and two of them are below. The other two seeds of this recipe pass 49 and 48, so a difference of one or two items is within seed noise. The exam is not part of the final promotion test. But it was one of the rules for keeping the 4000-update recipe and for judging the follow-up runs, so it is selection-exposed.</p>
      </div>
    {/if}
  </section>

  <section id="limits">
    {@render eyebrow('limits')}
    <h2>Where it still goes wrong</h2>
    <div class="case">
      <h3>Deflecting Swat: the labels favour the wrong neighbour</h3>
      <p>Deflecting Swat redirects a spell, and it is free if you control a commander. The ideal neighbour is Redirect. But the fine-tuned model ranks Obscuring Haze higher, and Haze shares only the free-with-a-commander line. The tags may be the reason. Swat shares three tags with Obscuring Haze and only two with Redirect, so training on tags may pull the wrong card closer. I did not isolate the cause, and my one follow-up run on effect-only labels did not fix it. On the trap exam, the margin went from {marginText(exPre, 'T001')} before tuning to {marginText(exDep, 'T001')} after. A margin below zero is a fail. In the explorer, pick Swat's redirect line under "Similar cards", and Redirect does show up.</p>
      <p><a href="{resolve('/')}?card=ae120613-97d6-4393-b39d-c3e6c076f5d6">See Deflecting Swat in the explorer</a></p>
    </div>
    <div class="case">
      <h3>Lexical twins still fool it</h3>
      <p>For Ghosts of the Innocent, the model still ranks Furnace of Rath above Kitsune Palliator (trap-exam margin {marginText(exDep, 'T021')}). I ran three follow-up runs, each with one intended change: more hard negatives, effect-only labels, and rare-tag weighting. They also used gradient checkpointing to save memory, so they are not fully controlled tests. None of them cleared its rule. Each also failed at least one exam item that its comparator passes. The rider problem is not solved.</p>
    </div>
    <ul class="plain">
      <li><b>Proxy, not people.</b> The scores and promotion tests measure how well the model agrees with community tags. I accepted the trap-exam items myself; a panel did not. No human relevance study has been run. I built a blinded pool of 1,256 card pairs from an earlier model. Nobody has labelled it, and its labelling tool is not in this repository.</li>
      <li><b>Selection-exposed.</b> The development scores helped me choose models. The only sealed result is for the earlier model. I fixed the seeds, thresholds and statistical test before the first training run, and recorded them in a private log that is not in this repository. The promotion test for the 4000-update recipe reuses those rules, but I wrote its design down after two of its three seeds had finished.</li>
      <li><b>Card to card only.</b> You always start from a card. Cognate is not a natural-language search model, and I never measured how well it handles typed queries.</li>
      <li><b>Exploratory views.</b> The explorer marks three views as exploratory: matches for one chosen ability, Line match and Face match. I do not claim that any of them is an improvement. With the earlier 2000-update model, Line match scored below Whole card on the development tag proxy (0.687 against 0.706). Face match scored about the same.</li>
      <li><b>Few seeds.</b> The three recipes in the main chart each ran with {seeds} seeds. The follow-up runs used one seed each. Three seeds say little about run-to-run variation.</li>
      <li><b>Long and rare cards.</b> The model reads at most {enc?.max_length} tokens, including its prefix. That cuts {n(TRUNCATED)} of the {n(DEV_UNIVERSE)} benchmark cards. Scores are lower for rare effects (0.636), long cards (0.628) and two-faced cards (0.650), against 0.714 overall.</li>
      <li><b>Lenient metric.</b> The top result almost always shares at least one tag with the card (MRR@10 is 0.985). So the metric mostly rewards putting good neighbours in the right order.</li>
      <li><b>Narrow comparison.</b> I compared only with the base model, a small BGE model and a TF-IDF baseline. I did not compare with larger embedding models.</li>
      <li><b>Display filter.</b> The explorer hides {n(KEYWORD_ONLY)} keyword-only cards, such as plain flyers, from the map and the lists. All scores include them.</li>
      <li><b>Snapshot.</b> Cards printed after the data snapshot are missing.</li>
    </ul>
  </section>

  <section id="built">
    {@render eyebrow('built')}
    <h2>Built with</h2>
    <p>Training and evaluation use Python, PyTorch and Hugging Face Transformers. The map uses scikit-learn (t-SNE). This is a static site built with SvelteKit, with no server of its own and no tracking. Card images load from Scryfall. A script fills in every result on this page from my local run and evaluation records. Those records are not in this repository.</p>
    <p>Card data and images come from <a href="https://scryfall.com" rel="noopener">Scryfall</a> and tags from <a href="https://tagger.scryfall.com" rel="noopener">Scryfall Tagger</a>. The idea of a trap exam, and several ideas for line-level labels and training, come from <a href="https://github.com/Ballchinian/Delvefall" rel="noopener">Delvefall</a>. I used the ideas only. I did not copy any code, data or exam items.</p>
    <p class="links"><a class="btn" href={REPO} rel="noopener">Code on GitHub</a> <a class="btn primary" href={resolve('/')}>Open the explorer</a></p>
  </section>
</article>

<style>
  /* one reading column; .bleed children widen to the outer track */
  .study {
    display: grid;
    grid-template-columns: [full-start] minmax(0, 1fr) [wide-start] minmax(0, 170px) [col-start] min(720px, 100%) [col-end] minmax(0, 170px) [wide-end] minmax(0, 1fr) [full-end];
    padding: 0 clamp(16px, 4vw, 32px) 56px;
  }
  .study > *, section > :global(*) { grid-column: col; }
  .study > section { grid-column: full; display: grid; grid-template-columns: subgrid; padding-top: 64px; scroll-margin-top: 56px; }
  .bleed, .study > .pipeline { grid-column: wide; }

  .hero { padding-top: clamp(32px, 8vh, 80px); }
  .hero h1 { margin: 0 0 18px; text-wrap: balance; }
  .lede { font-size: 1.12rem; color: var(--muted); max-width: 62ch; }
  .hero-actions { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 26px 0 8px; }
  .hero-search { flex: 1 1 280px; }
  .hero-actions .btn { height: 48px; padding: 0 20px; font-size: 0.95rem; }
  .short { margin-top: 28px; padding: 16px 20px 6px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); }
  .short h2 { margin: 0 0 8px; font-size: 1.05rem; }
  .short ul { margin: 0; padding-left: 1.2em; }
  .short li { margin: 0 0 8px; max-width: 70ch; }
  .facts { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 30px 0 0; }
  .facts div { padding: 10px 2px 0; border-top: 2px solid var(--ink); }
  .facts dt { font-size: 0.76rem; color: var(--muted); }
  .facts dd { margin: 2px 0 0; font-size: 1.35rem; font-weight: 600; }

  .pipeline { margin-top: 44px; }
  .pipeline ol { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); margin: 0; padding: 0; list-style: none; }
  .pipeline li { position: relative; }
  /* the line joining the step numbers */
  .pipeline li + li::before { content: ''; position: absolute; top: 21px; left: -50%; right: 50%; height: 1px; background: var(--line); margin: 0 18px; }
  .pipeline a { display: grid; justify-items: center; gap: 2px; padding: 8px 4px; text-align: center; text-decoration: none; border-radius: var(--radius); }
  .pipeline a:hover .no { background: var(--ink); color: var(--bg); }
  .no { display: grid; place-items: center; width: 26px; height: 26px; margin-bottom: 6px; border: 1px solid var(--ink); border-radius: 50%; background: var(--bg); font-size: 0.78rem; font-weight: 600; }
  .pipeline b { font-size: 0.86rem; font-weight: 650; line-height: 1.2; }
  .pipeline small { font-size: 0.74rem; color: var(--muted); line-height: 1.25; }

  .eyebrow { margin: 0 0 8px; font-size: 0.78rem; font-weight: 600; color: var(--muted); }
  .eyebrow .num { display: inline-grid; place-items: center; width: 20px; height: 20px; margin-right: 4px; border: 1px solid currentColor; border-radius: 50%; font-size: 0.7rem; }
  h2 { margin: 0 0 16px; text-wrap: balance; }
  .sub { margin: 40px 0 8px; font-size: 1.12rem; }
  h2 + .sub { margin-top: 8px; }
  .small { font-size: 0.84rem; }

  .pair { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 6px 0 22px; }
  .quote { margin: 0; padding: 14px 16px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); }
  .quote figcaption { font: 600 1rem var(--sans); margin-bottom: 6px; }
  .quote figcaption span { display: block; font: 400 0.76rem var(--sans); color: var(--muted); }
  .quote p { margin: 0; font-family: var(--serif); font-size: 0.95rem; line-height: 1.5; }
  .quote.good { border-color: var(--wubrg-g); }
  mark { background: var(--mark); color: inherit; padding: 0 2px; }

  .splits { margin: 4px 0 18px; }
  .splitbar { display: flex; gap: 2px; height: 14px; }
  .splitbar span:first-child { border-radius: 3px 0 0 3px; }
  .splitbar span:last-child { border-radius: 0 3px 3px 0; }
  .fit { background: var(--ink); }
  .dev { background: var(--wubrg-c); }
  .sealed { background: repeating-linear-gradient(135deg, var(--muted) 0 2px, transparent 2px 5px); box-shadow: inset 0 0 0 1px var(--muted); }
  .splits ul { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin: 14px 0 0; padding: 0; list-style: none; font-size: 0.88rem; }
  .splits li i { display: inline-block; width: 10px; height: 10px; margin-right: 6px; border-radius: 2px; vertical-align: 0; }
  .splits li .num { color: var(--muted); }
  .splits li > span:last-child { display: block; margin-top: 2px; color: var(--muted); font-size: 0.84rem; }

  .flow { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 28px; margin: 4px 0 22px; padding: 0; list-style: none; counter-reset: none; }
  .flow li { position: relative; padding: 12px 14px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); font-size: 0.9rem; overflow-wrap: anywhere; }
  .flow li + li::before { content: '→'; position: absolute; left: -22px; top: 50%; transform: translateY(-50%); color: var(--muted); }
  .flow li.accent { border-color: var(--ink); }
  .flow small, .t small { display: block; font: 600 0.7rem var(--sans); color: var(--muted); margin-bottom: 4px; }
  .flow span { display: block; color: var(--muted); font-size: 0.82rem; }
  .cols2 { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr); gap: 24px; align-items: start; margin-bottom: 8px; }
  .triplet { margin: 0; display: grid; gap: 8px; }
  .t { padding: 10px 14px; border-radius: var(--radius); border: 1px solid var(--line); background: var(--surface); font: 600 0.95rem var(--sans); }
  .t.anchor { border-color: var(--ink); }
  .t.pos { border-left: 4px solid var(--wubrg-g); margin-left: 20px; }
  .t.neg { border-left: 4px solid var(--wubrg-r); margin-left: 20px; }

  .caveat { margin-top: 16px; padding: 10px 14px; border-left: 3px solid var(--muted); background: var(--surface); font-size: 0.9rem; }
  .exam { display: grid; gap: 12px; }
  .meters { display: grid; gap: 8px; }
  .meter { display: grid; grid-template-columns: 96px 1fr 52px; gap: 12px; align-items: center; font-size: 0.84rem; }
  .ml { color: var(--muted); }
  .cells { display: grid; grid-template-columns: repeat(auto-fill, minmax(6px, 1fr)); gap: 2px; }
  .cells i { height: 14px; border-radius: 1px; box-shadow: inset 0 0 0 1px var(--wubrg-c); }
  .cells i.on { background: var(--wubrg-c); box-shadow: none; }
  .meter:last-child .cells i.on { background: var(--ink); }
  .meter b { font-size: 1.05rem; text-align: right; }
  .meter b span { font-size: 0.8rem; font-weight: 400; color: var(--muted); }

  .case { margin: 4px 0 22px; }
  .case h3 { margin-bottom: 6px; }
  .plain { margin: 8px 0 0; padding-left: 1.1em; }
  .plain li { margin: 8px 0; }
  .links { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }

  @media (max-width: 860px) {
    .pipeline ol { grid-template-columns: repeat(4, minmax(0, 1fr)); row-gap: 8px; }
    .pipeline li:nth-child(5)::before { display: none; }
  }
  @media (max-width: 640px) {
    .facts { grid-template-columns: repeat(2, minmax(0, 1fr)); row-gap: 18px; }
    .pipeline ol { grid-template-columns: minmax(0, 1fr); }
    .pipeline li::before { display: none; }
    .pipeline a { grid-template-columns: 26px auto 1fr; justify-items: start; align-items: center; gap: 10px; text-align: left; padding: 6px 0; }
    .no { margin: 0; }
    .pipeline small { justify-self: end; text-align: right; }
    .pair, .cols2, .splits ul, .flow { grid-template-columns: minmax(0, 1fr); }
    .flow { gap: 22px; }
    .flow li + li::before { content: '↓'; left: 50%; top: -20px; transform: translateX(-50%); }
    .meter { grid-template-columns: 72px 1fr 44px; gap: 8px; }
  }
</style>
