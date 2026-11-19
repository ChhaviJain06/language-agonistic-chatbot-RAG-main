const STARTERS: { tag: string; emoji: string; q: string }[] = [
  { tag: "English", emoji: "🏛️", q: "Who was Ashoka and why did he embrace Buddhism?" },
  { tag: "हिन्दी", emoji: "📜", q: "मौर्य साम्राज्य की स्थापना किसने की थी?" },
  { tag: "Hinglish", emoji: "⚔️", q: "Akbar ne Haldighati ka yudh kiske khilaaf lada tha?" },
  { tag: "Multi-hop", emoji: "🔗", q: "Who was the grandfather of the emperor who built the Taj Mahal?" },
  { tag: "False premise", emoji: "🧐", q: "Why did Ashoka build the Taj Mahal?" },
  { tag: "தமிழ்", emoji: "🌴", q: "ராஜராஜ சோழன் கட்டிய கோயில் எது?" },
];

export default function EmptyState({ onPick, disabled }: { onPick: (q: string) => void; disabled: boolean }) {
  return (
    <div className="hero">
      <div className="hero-mark" aria-hidden="true">भ</div>
      <h1>Ask India&rsquo;s history, in your language</h1>
      <p className="hero-sub">
        BHASHA answers from Wikipedia sources with citations, in English, Hindi, Hinglish, Marathi, Bengali and Tamil.
        Type below or press <kbd>/</kbd> to focus the box.
      </p>
      <div className="starters" role="list">
        {STARTERS.map((s) => (
          <button
            key={s.q}
            type="button"
            role="listitem"
            className="starter"
            disabled={disabled}
            onClick={() => onPick(s.q)}
          >
            <span className="starter-tag">
              <span aria-hidden="true">{s.emoji}</span> {s.tag}
            </span>
            <span className="starter-q">{s.q}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
