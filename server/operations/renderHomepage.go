package operations

import (
	"html/template"
	"net/http"
)

func RenderHomePage(w http.ResponseWriter, r *http.Request) {
	template := template.Must(template.New("photo_gallery.html").ParseGlob("templates/photo_gallery.html"))
	template.Execute(w, nil)
}
