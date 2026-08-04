document.querySelectorAll('.comparison').forEach((comparison) => {
  const range = comparison.querySelector('input[type="range"]');

  const updatePosition = () => {
    comparison.style.setProperty('--position', `${range.value}%`);
  };

  range.addEventListener('input', updatePosition);
  updatePosition();
});

document.querySelectorAll('.author-placeholder').forEach((link) => {
  link.addEventListener('click', (event) => event.preventDefault());
});
