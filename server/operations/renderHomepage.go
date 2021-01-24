package operations

import (
	"github.com/howardwang15/surveillance/server/db"
	"github.com/howardwang15/surveillance/server/models"
	"html/template"
	"net/http"
	"path"
	"strconv"
	"time"
)

func formatTime(ts time.Time) string {
	layout := "1/2/2006 3:04:05 PM"
	return ts.Format(layout)
}

type VideoResponse struct {
	Time      string
	ImageFile string
	VideoFile string
	ID        uint64
}

type Response struct {
	Videos []VideoResponse
	NumPages int
	Page int
}

func RenderHomePage(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	page := query["page"]
	pageFetch := 1

	if len(page) != 0 {
		var err error
		pageFetch, err = strconv.Atoi(page[0])
		if err != nil {
			pageFetch = 1
		}
	}

	var videos []models.TempVideo
	NUM_RESULTS := 20
	var count int64
	db.DB.Model(&models.TempVideo{}).Count(&count)
	numPages := int(count) / NUM_RESULTS

	if pageFetch > numPages {
		pageFetch = numPages
	}
	db.DB.Order("Id desc").Limit(NUM_RESULTS).Offset((pageFetch - 1) * NUM_RESULTS).Find(&videos)

	videoResponses := make([]VideoResponse, len(videos))
	for i := 0; i < len(videos); i++ {
		videoResponses[i] = VideoResponse{formatTime(videos[i].StartTime),
			path.Join("assets", videos[i].FirstFrame),
			path.Join("assets", videos[i].VideoName),
			videos[i].Id,
		}
	}

	template, err := template.New("photo_gallery.html").Funcs(template.FuncMap{
		"htmlSafe": func(html string) template.HTML {
			return template.HTML(html)
		},
		"add": func(a int, b int) int {
			return a + b
		},
		"sub": func(a int, b int) int {
			return a - b
		},
	}).ParseFiles("templates/photo_gallery.html")

	if err != nil {
		panic(err)
	}
	response := Response{videoResponses, numPages, pageFetch}
	template.Execute(w, response)
}
