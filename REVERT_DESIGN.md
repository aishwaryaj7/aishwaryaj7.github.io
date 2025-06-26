# How to Revert to Original Design

If you want to revert back to the original design, follow these steps:

## Quick Revert Commands

```bash
# 1. Restore original mkdocs.yml
cp mkdocs.yml.backup mkdocs.yml

# 2. Remove custom CSS (optional - you can keep it for future use)
# rm docs/stylesheets/extra.css

# 3. Remove modern assets
rm -rf docs/assets/images/

# 4. Remove GitHub Pages workflow (optional)
# rm .github/workflows/deploy-portfolio.yml

# 5. Restore original content files (if you have backups)
# Note: The new design preserves all your original content
# Only the layout and styling have changed

# 6. Test the reverted site
mkdocs serve
```

## What Was Changed

### Files Modified:
- `mkdocs.yml` - Enhanced with Material theme features
- `docs/index.md` - Added hero section and modern layout
- `docs/contact.md` - Updated with card-based design
- `docs/projects/index.md` - Added project cards
- `docs/stylesheets/extra.css` - Enhanced with modern styling

### Files Added:
- `docs/assets/images/favicon.svg` - Custom favicon
- `docs/assets/images/logo.svg` - Custom logo
- `.github/workflows/deploy-portfolio.yml` - GitHub Pages deployment
- `overrides/` directory - For future customizations

### Original Content Preserved:
- All your markdown content is preserved
- Blog posts remain unchanged
- Project documentation intact
- Resume content maintained

## Selective Revert Options

### Keep Modern Styling, Revert Layout:
```bash
# Only revert the layout changes, keep the styling
cp mkdocs.yml.backup mkdocs.yml
# Keep docs/stylesheets/extra.css for future use
```

### Keep Layout, Revert Styling:
```bash
# Keep the modern layout, use original styling
rm docs/stylesheets/extra.css
# Edit mkdocs.yml to remove extra_css section
```

### Keep Everything, Just Disable Features:
Edit `mkdocs.yml` and comment out specific features you don't want:
```yaml
# Comment out features you don't want
# - navigation.tabs
# - navigation.instant
# etc.
```

## Backup Information

- Original mkdocs.yml backed up as `mkdocs.yml.backup`
- All original content preserved in the new design
- No data loss - only presentation changes

## Support

If you need help with the revert process or want to customize specific aspects, the changes are modular and can be selectively applied or removed.
