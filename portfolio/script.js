/* ===================================================
   PORTFOLIO SCRIPTS — Muhammad Farooq
   =================================================== */

// Nav scroll effect
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  if (window.scrollY > 20) navbar.classList.add('scrolled');
  else navbar.classList.remove('scrolled');
});

// Hamburger mobile menu
const hamburger = document.getElementById('hamburger');
const mobileMenu = document.getElementById('mobileMenu');
hamburger.addEventListener('click', () => {
  mobileMenu.classList.toggle('open');
});
mobileMenu.querySelectorAll('a').forEach(link => {
  link.addEventListener('click', () => mobileMenu.classList.remove('open'));
});

// Fade-in on scroll
const observer = new IntersectionObserver(
  entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        observer.unobserve(e.target);
      }
    });
  },
  { threshold: 0.12 }
);

document.querySelectorAll(
  '.about-grid, .stack-card, .project-card, .timeline-item, .contact-inner'
).forEach(el => {
  el.classList.add('fade-in');
  observer.observe(el);
});

// Staggered card animation
document.querySelectorAll('.stack-grid .stack-card').forEach((card, i) => {
  card.style.transitionDelay = `${i * 60}ms`;
});
document.querySelectorAll('.projects-grid .project-card').forEach((card, i) => {
  card.style.transitionDelay = `${i * 60}ms`;
});
