FROM jonasal/nginx-certbot:5.1.1-nginx1.26.0

# create /var/www/letsencrypt for temp file
RUN mkdir -p /var/www/letsencrypt/.well-known/acme-challenge

# permission
RUN chown -R nginx:nginx /var/www/letsencrypt
RUN chmod -R 755 /var/www/letsencrypt

COPY ./*.conf /etc/nginx/user_conf.d
