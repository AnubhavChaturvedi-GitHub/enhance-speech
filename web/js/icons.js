// Inline icons, drawn for this project.
const svg = (body, viewBox = "0 0 22 22") =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;

export const icons = {
  play: svg('<path d="M5 4.9c0-1.3 1.5-2.1 2.7-1.5l10 6.1c1.1.7 1.1 2.3 0 3l-10 6.1C6.5 19.2 5 18.4 5 17.1z"/>'),
  pause: svg('<rect x="4" y="3" width="5" height="16" rx="2.5"/><rect x="13" y="3" width="5" height="16" rx="2.5"/>'),
  trash: svg('<path d="M3.5 5.5h15M8.5 5.5V4a1 1 0 0 1 1-1h3a1 1 0 0 1 1 1v1.5M5.5 5.5l.8 12a1.5 1.5 0 0 0 1.5 1.5h6.4a1.5 1.5 0 0 0 1.5-1.5l.8-12M9.5 9.5v5.5M12.5 9.5v5.5"/>'),
  info: svg('<circle cx="11" cy="11" r="8.5"/><path d="M11 10.5v5"/><circle cx="11" cy="7" r=".6" fill="currentColor"/>'),
  download: svg('<path d="M11 3v11M6.5 9.5 11 14l4.5-4.5M3.5 18.5h15"/>'),
  volume: svg('<path d="M3 8.5v5h3.5l5 4v-13l-5 4z"/><path d="M15 8a4 4 0 0 1 0 6M17.5 5.5a7.5 7.5 0 0 1 0 11"/>'),
  muted: svg('<path d="M3 8.5v5h3.5l5 4v-13l-5 4z"/><path d="m15.5 8.5 5 5M20.5 8.5l-5 5"/>'),
  check: svg('<path d="m4 11.5 4.5 4.5L18 6.5"/>'),
  text: svg('<path d="M3.5 5.5h15M3.5 11h15M3.5 16.5h9"/>'),
  captions: svg('<rect x="2.5" y="4.5" width="17" height="13" rx="2.5"/><path d="M9.5 9.5a2.2 2.2 0 1 0 0 3M16 9.5a2.2 2.2 0 1 0 0 3"/>'),
  mic: svg('<rect x="8" y="2.5" width="6" height="11" rx="3"/><path d="M4.5 10.5a6.5 6.5 0 0 0 13 0M11 17v2.5"/>'),
  video: svg('<rect x="2.5" y="5" width="12" height="12" rx="2.5"/><path d="m14.5 9.5 5-3v9l-5-3"/>'),
  layers: svg('<rect x="7" y="7" width="12" height="12" rx="2.5"/><path d="M3 15V5a2 2 0 0 1 2-2h10M13 10.5v5M10.5 13h5"/>'),
  sliders: svg('<path d="M3 6h9M16 6h3M3 16h3M10 16h9"/><circle cx="14" cy="6" r="2"/><circle cx="8" cy="16" r="2"/>'),
  infinity: svg('<path d="M6.5 15.5a4.5 4.5 0 0 1 0-9c3.5 0 5.5 9 9 9a4.5 4.5 0 0 0 0-9c-3.5 0-5.5 9-9 9z"/>'),
  lock: svg('<rect x="4" y="10" width="14" height="9.5" rx="2.5"/><path d="M7.5 10V7a3.5 3.5 0 0 1 7 0v3"/>'),
  wifiOff: svg('<path d="m3 3 16 16M8.5 16.5a3.5 3.5 0 0 1 5 0M5 12.5a9 9 0 0 1 4-2.3M17 12.5a9 9 0 0 0-3-2M2 8.5a13 13 0 0 1 3.5-2.4M20 8.5a13 13 0 0 0-10-3.2"/>'),
  folder: svg('<path d="M2.5 6a2 2 0 0 1 2-2h4l2 2.5h7a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>'),
  sparkle: svg('<path d="M11 2.5 13 9l6.5 2-6.5 2-2 6.5L9 13l-6.5-2L9 9z"/>'),
};

export const dropzoneArt = `
<svg class="dropzone-icon" viewBox="0 0 88 75" fill="none" aria-hidden="true">
  <rect x="24" y="21" width="44" height="44" rx="9" fill="#fff" stroke="#222" stroke-width="2.4"/>
  <rect x="33" y="30" width="44" height="44" rx="9" fill="#edeeff" stroke="#222" stroke-width="2.4" transform="translate(-4 -4)"/>
  <g fill="#5c5ce0"><rect x="37" y="42" width="3" height="12" rx="1.5"/><rect x="43" y="36" width="3" height="24" rx="1.5"/><rect x="49" y="39" width="3" height="18" rx="1.5"/><rect x="55" y="42" width="3" height="12" rx="1.5"/><rect x="61" y="45" width="3" height="6" rx="1.5"/></g>
  <circle cx="20" cy="17" r="11" fill="#fff" stroke="#5c5ce0" stroke-width="2.2" stroke-dasharray="3.5 3.5"/>
  <path d="M20 12v10M15 17h10" stroke="#222" stroke-width="2.2" stroke-linecap="round"/>
</svg>`;

export const emptyArt = `
<svg viewBox="0 0 123 123" fill="none" aria-hidden="true">
  <circle cx="61.5" cy="61.5" r="61.5" fill="#edeeff"/>
  <g fill="#9599ff"><rect x="30" y="52" width="6" height="19" rx="3"/><rect x="42" y="40" width="6" height="43" rx="3"/><rect x="54" y="30" width="6" height="63" rx="3"/><rect x="66" y="42" width="6" height="39" rx="3"/><rect x="78" y="36" width="6" height="51" rx="3"/><rect x="90" y="52" width="6" height="19" rx="3"/></g>
</svg>`;

export const bulkArt = `
<svg viewBox="0 0 156 132" fill="none" aria-hidden="true">
  <rect x="14" y="30" width="86" height="86" rx="16" fill="#e0e2ff" stroke="#fff" stroke-width="3" transform="rotate(-16 57 73)"/>
  <rect x="30" y="22" width="86" height="86" rx="16" fill="#edeeff" stroke="#fff" stroke-width="3" transform="rotate(-7 73 65)"/>
  <rect x="50" y="14" width="86" height="86" rx="16" fill="#fff" stroke="#d3d5ff" stroke-width="2.5"/>
  <g fill="#5c5ce0"><rect x="72" y="50" width="4" height="14" rx="2"/><rect x="80" y="42" width="4" height="30" rx="2"/><rect x="88" y="36" width="4" height="42" rx="2"/><rect x="96" y="42" width="4" height="30" rx="2"/><rect x="104" y="50" width="4" height="14" rx="2"/></g>
</svg>`;
