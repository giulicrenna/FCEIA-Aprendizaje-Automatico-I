# Aprendizaje Automático I — Sitio de la cátedra

Sitio web oficial de la materia **Aprendizaje Automático 1** de la Facultad de Ciencias Exactas, Ingeniería y Agrimensura (FCEIA) — Universidad Nacional de Rosario (UNR).

Construido con [Jekyll](https://jekyllrb.com/) y el tema [just-the-docs](https://just-the-docs.github.io/just-the-docs/), desplegado en [GitHub Pages](https://pages.github.com/).

**Docentes:** Joel Spak · Giuliano Crenna

## Previsualización local

Requisitos: Ruby 3.x y Bundler.

```bash
bundle install
bundle exec jekyll serve
```

El sitio queda disponible en <http://localhost:4000>.

## Despliegue en GitHub Pages

1. Pusheá los cambios a la rama `main`.
2. En GitHub, ir a **Settings → Pages**.
3. En **Source**, elegir **Deploy from a branch**.
4. Branch: `main`, folder: `/ (root)`.
5. Guardar. GitHub construye y publica en la URL que aparece arriba.

> ⚠️ Configurar `url` y `baseurl` en [`_config.yml`](_config.yml) con el usuario y nombre de repo reales antes del primer deploy.

## Estructura del repositorio

- [`_config.yml`](_config.yml) — Configuración general (tema, plugins, idioma, URL).
- [`assets/css/custom.scss`](assets/css/custom.scss) — Paleta institucional FCEIA/UNR.
- [`index.md`](index.md) — Página de inicio (única página de contenido por ahora).
- [`Gemfile`](Gemfile) — Dependencias para desarrollo local.

## Cómo agregar páginas de contenido

Las páginas se agregan como archivos `.md` (o `.html`) en la raíz del repo. Cada página necesita front matter para aparecer en el sidebar de just-the-docs:

```markdown
---
layout: default
title: Programa
nav_order: 2
---

# Contenido de la página
```

El campo `nav_order` controla el orden en el sidebar. Para agrupar páginas bajo una sección padre, agregá `parent: NombrePadre` en el front matter de la página hija.

Secciones típicas que suelen necesitarse en una cátedra:
- Programa
- Cronograma
- Clases teóricas
- Trabajos prácticos
- Evaluación
- Bibliografía
- Recursos
