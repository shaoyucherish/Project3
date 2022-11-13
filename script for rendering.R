#Code for rendering file 
rmarkdown::render("project3.Rmd", 
                  output_format = "github_document",
                  output_file = "README.md",
                  output_options = list(toc=TRUE, toc_depth=1, toc_float=TRUE))