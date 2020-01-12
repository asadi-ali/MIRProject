import json
import scrapy


class PapersSpider(scrapy.Spider):
    name = "papers"

    def __init__(self, number_of_papers=5000):
        super().__init__()
        self.papers = set()
        self.number_of_papers = int(number_of_papers)

    def start_requests(self):
        with open('start.txt') as f:
            urls = f.readlines()
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        paper_id = response.url.split("/")[-1]
        if paper_id in self.papers:
            return

        if len(self.papers) >= self.number_of_papers:
            return
        self.papers.add(paper_id)
        self.log(len(self.papers))

        data = {
            'id': paper_id,
            'title': response.xpath('//meta[@name="citation_title"]/@content').get(),
            'authors': response.xpath('//meta[@name="citation_author"]/@content').getall(),
            'abstract': response.xpath('//meta[@name="description"]/@content').get(),
            'date': response.xpath('//meta[@name="citation_publication_date"]/@content').get(),
            'references': list(map(lambda x: x.split('/')[-1], response.css('div.references div.paper-citation div h2 a::attr("href")').getall())),
        }

        with open("papers/%s.txt" % paper_id, 'w') as f:
            f.write(json.dumps(data))

        for next_url in response.css('div.references div.paper-citation div h2 a::attr("href")').getall()[:5]:
            yield response.follow(next_url, callback=self.parse)
