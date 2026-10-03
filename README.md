# AI 学习课程

中文 AI 互动课件。打开 `index.html` 即可浏览全部课程。

这个仓库只包含现成网页和页面引用的下载资源。配套 `.py` 文件是供学习者下载的练习代码；网站不执行这些文件。

## Cloudflare Pages

连接此仓库并设置：

| 配置项 | 值 |
| --- | --- |
| Production branch | `main` |
| Framework preset | None |
| Root directory | 留空 |
| Build command | 留空 |
| Build output directory | `.` |

无需构建或安装依赖。若已有 Pages 项目，请清除之前的 `pages` 根目录、`sh build.sh` 或 `sh scripts/build-site.sh` 构建命令，以及 `dist` 输出目录，改为上表中的设置。
