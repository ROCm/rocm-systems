# Daily sync - GL2 residency on gfx1250

~40 seconds.

---

While measuring the non-temporal blit copy, I found that **gfx1250 does not retain GL2 across
a kernel dispatch**. I think that is the limiting factor on why the hint barely helps - the
whole point of a non-temporal hint is to stop the copy evicting what the next kernel needs,
and nothing survives to the next kernel anyway. So only genuinely concurrent work can benefit.

I ruled out the three obvious causes. It is not the gfx12 system-scope acquire fence - MI450
does not even take that code path. It is not fence scope at all - forcing system scope on
every dispatch changes nothing. And it is not the memory type - I tested six allocation kinds
and none of them retain.

To be fair to the docs, a full flush is allowed - the spec says the CP may do either a
selective NC invalidate or a full writeback-invalidate. What the docs do not explain is that
it happens regardless of fence scope, when they tie it to system scope only. So the question
is whether this is deliberate on MI450.

Chasing that turned up a second thing. GL2 actually measures somewhere between ninety-six and
a hundred and twenty-eight megabytes, not the four the driver reports - and the KFD record
behind that figure has all its geometry fields zeroed, so it is an unpopulated stub rather than
a wrong number. That mattered, because I had sized my concurrency test against the four
megabyte figure. Redoing it with a properly sized working set, the benefit is up to five
percent, not one. So the change is worth more than I said on Tuesday.

**What I need:** to ask CP or firmware whether it is expected that GL2 is not retained across a
kernel dispatch. If it is not expected, recovering that residency is a much bigger lever than
my ticket.

---

**If asked - is the blit change still worth taking?** Yes, though narrower than I first said.
The isolated copy only gains in the 96 to 128 meg band, about four percent, and it is flat
noise everywhere else. The real case is concurrency: 2.4 to 4.8 percent off a co-running
kernel's runtime, peaking near 32 meg working set. No regression anywhere, default off.

**If asked - could the measurement be wrong?** Cold and warm are within one percent where I
would expect two-times. Four laps inside one kernel are twice as fast, so caching works - it
just does not survive the boundary. I also re-ran everything with the flush swept from 128 meg
to two gig, in case my cold case was not actually cold. It made no difference, which is itself
consistent - the dispatch boundary had already done the flushing for me.

**If asked - did anything else change when you re-measured?** Two things I had reported are
withdrawn. The one-percent concurrency figure was a victim sized against the wrong cache. And
the NT_RT variant's one-percent win turned out to be codegen from hand-writing the store, not
the temporal hint - the control for that sits right on the hint's effect size, so I had been
crediting the wrong thing.

Written up in REPORT.md; the residency finding has its own write-up in
FINDING-gl2-residency.md, and everything withdrawn is listed in CHANGELOG.md.
