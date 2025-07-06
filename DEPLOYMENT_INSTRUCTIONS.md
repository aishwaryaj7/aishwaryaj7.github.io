# 🚀 GitHub Pages Deployment Instructions

## 📋 What I've Set Up For You

✅ **Updated GitHub Actions workflow** (`.github/workflows/deploy.yml`)
✅ **Created requirements.txt** for dependencies
✅ **Configured automatic deployment** on push to main branch

## 🎯 Your Next Steps

### **Step 1: Push Changes to GitHub**

```bash
# Add all changes
git add .

# Commit changes
git commit -m "Set up GitHub Pages deployment with updated portfolio"

# Push to main branch
git push origin main
```

### **Step 2: Enable GitHub Pages (One-time setup)**

1. **Go to your repository** on GitHub: `https://github.com/aishwaryaj7/aishwaryaj7.github.io`

2. **Click on "Settings"** tab

3. **Scroll down to "Pages"** in the left sidebar

4. **Under "Source"** select:
   - Source: **"GitHub Actions"**
   - (Don't select "Deploy from a branch")

5. **Save the settings**

### **Step 3: Wait for Deployment**

- After pushing, GitHub Actions will automatically build and deploy your site
- Check the **"Actions"** tab to see the deployment progress
- First deployment takes 2-5 minutes

### **Step 4: Access Your Live Portfolio**

Your portfolio will be available at:
```
https://aishwaryaj7.github.io
```

## 🔧 What the Workflow Does

1. **Triggers** on every push to main branch
2. **Installs** Python and MkDocs dependencies
3. **Builds** your MkDocs site
4. **Deploys** to GitHub Pages automatically
5. **Caches** dependencies for faster builds

## 🎨 Features Enabled

- ✅ **Material Design** theme
- ✅ **Mermaid diagrams** support
- ✅ **Code highlighting** with Pygments
- ✅ **Image lightbox** with Glightbox
- ✅ **Responsive design** for mobile/desktop
- ✅ **Fast loading** with optimized assets

## 🔄 Future Updates

Every time you make changes:
1. **Edit your files** locally
2. **Commit and push** to main branch
3. **GitHub Actions** automatically rebuilds and deploys
4. **Changes go live** in 2-3 minutes

## 🌐 Custom Domain (Optional)

If you want a custom domain like `aishwaryajauhari.com`:

1. **Buy a domain** from Namecheap, GoDaddy, etc.
2. **Add CNAME file** to your repository root:
   ```
   echo "yourdomain.com" > CNAME
   ```
3. **Configure DNS** at your domain provider:
   - Add CNAME record pointing to `aishwaryaj7.github.io`
4. **Update GitHub Pages settings** with your custom domain

## 📱 For LinkedIn & Resume

Once deployed, add this URL to:
- **LinkedIn**: Profile → Contact Info → Website
- **Resume**: Header or contact section
- **Business cards**: Professional portfolio link

## 🆘 Troubleshooting

### If deployment fails:
1. Check the **Actions** tab for error messages
2. Ensure all files are committed and pushed
3. Verify `mkdocs.yml` syntax is correct

### If site doesn't load:
1. Wait 5-10 minutes for DNS propagation
2. Try accessing in incognito/private browsing
3. Check GitHub Pages settings are correct

## 🎉 Success Indicators

✅ **Green checkmark** in Actions tab
✅ **Site loads** at `https://aishwaryaj7.github.io`
✅ **All pages work** (Projects, Blog, Resume, Contact)
✅ **Images display** correctly
✅ **Navigation works** smoothly

---

**🚀 Ready to deploy? Just push your changes and watch the magic happen!**
